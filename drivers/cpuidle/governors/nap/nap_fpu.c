// SPDX-License-Identifier: GPL-2.0
/*
 * nap_fpu.c — FPU/SIMD code for the NAP cpuidle governor
 *
 * This file is compiled with FPU/SSE flags enabled (CC_FLAGS_FPU).
 * ALL functions here MUST be called only from within
 * kernel_fpu_begin()/kernel_fpu_end() blocks.
 *
 * Keeping FPU code in a separate translation unit ensures the compiler
 * cannot emit SSE/x87 instructions in non-FPU code paths (nap.c),
 * which would silently corrupt userspace FPU register state.
 */

#include <linux/cpuidle.h>
#include <linux/kernel_stat.h>
#include <linux/math64.h>
#include <linux/percpu.h>
#include <linux/sched/clock.h>
#include <linux/sched/stat.h>
#include <linux/string.h>
#include <linux/tick.h>

#include "nap.h"

/* ================================================================
 * Float math helpers
 * ================================================================ */

static inline float float_min(float a, float b) { return a < b ? a : b; }
static inline float float_max(float a, float b) { return a > b ? a : b; }

/* Scalar log2 approximation (same algorithm as fast_log2f_sse) */
static inline float fast_log2f(float x)
{
	union { float f; u32 i; } u = { .f = x };
	int exp = (int)((u.i >> 23) & 0xFFu) - 127;
	float e = (float)exp;
	float m, p;

	u.i = (u.i & 0x7FFFFFu) | (127u << 23);
	m = u.f - 1.0f;

	p = m * 0.4808f;
	p = 0.7213f - p;
	p = m * p;
	p = 1.4425f - p;
	p = m * p;

	return e + p;
}

/* ================================================================
 * Deterministic PRNG for weight initialization (LCG)
 * ================================================================ */

static inline float nap_prng_float(u32 *state)
{
	*state = *state * 1664525u + 1013904223u;
	return (float)(s32)*state * (1.0f / 2147483648.0f);
}

/* ================================================================
 * ISA dispatch via static keys
 * ================================================================ */

static inline void nap_nn_forward(const float *input, float *output,
				  float *hidden_save,
				  const struct nap_weights *w)
{
	if (static_branch_unlikely(&nap_use_avx512))
		nap_nn_forward_avx512(input, output, hidden_save, w);
	else if (static_branch_unlikely(&nap_use_avx2))
		nap_nn_forward_avx2(input, output, hidden_save, w);
	else
		nap_nn_forward_sse2(input, output, hidden_save, w);
}

static inline void nap_nn_learn(struct nap_cpu_data *d, int ideal)
{
	if (static_branch_unlikely(&nap_use_avx512))
		nap_nn_learn_avx512(d, ideal);
	else if (static_branch_unlikely(&nap_use_avx2))
		nap_nn_learn_avx2(d, ideal);
	else
		nap_nn_learn_sse2(d, ideal);
}

/* ================================================================
 * Weight initialization
 *
 * Hidden layer:  Xavier uniform init with fixed PRNG seed (deterministic).
 * Output biases: informed by per-state exit_latency_ns so that
 *                shallow (low-latency) states are preferred initially.
 *                The NN then learns to override these via online training.
 * ================================================================ */

#define NAP_PRNG_SEED 42u

static void nap_init_weights(struct nap_weights *w,
			     struct cpuidle_driver *drv)
{
	u32 rng = NAP_PRNG_SEED;
	float scale_h1, scale_out;
	int i, j;

	/* Xavier uniform: U(-sqrt(6/(fan_in+fan_out)), +sqrt(6/(...))) */
	scale_h1  = __builtin_sqrtf(6.0f / (float)(NAP_INPUT_SIZE + NAP_HIDDEN_SIZE));
	scale_out = __builtin_sqrtf(6.0f / (float)(NAP_HIDDEN_SIZE + NAP_OUTPUT_SIZE));

	/* Hidden layer weights */
	for (i = 0; i < NAP_INPUT_SIZE; i++)
		for (j = 0; j < NAP_HIDDEN_SIZE; j++)
			w->w_h1[i][j] = nap_prng_float(&rng) * scale_h1;

	/* Hidden biases: zero (standard) */
	memset(w->b_h1, 0, sizeof(w->b_h1));

	/* Output layer weights (zero padding columns for SIMD) */
	memset(w->w_out, 0, sizeof(w->w_out));
	for (j = 0; j < NAP_HIDDEN_SIZE; j++)
		for (i = 0; i < NAP_OUTPUT_SIZE; i++)
			w->w_out[j][i] = nap_prng_float(&rng) * scale_out;

	/* Output biases: -0.1 * log2(exit_latency_ns) per state.
	 * Shallow states get ~0, deep states get ~-1.7.
	 * Unused/padding slots get -100 to ensure they're never selected.
	 */
	for (i = 0; i < NAP_OUTPUT_STRIDE; i++) {
		if (drv && i < drv->state_count) {
			float lat = float_max(
				(float)drv->states[i].exit_latency_ns, 1.0f);
			w->b_out[i] = -fast_log2f(lat) * 0.1f;
		} else {
			w->b_out[i] = -100.0f;
		}
	}
}

/* ================================================================
 * Feature extraction helpers
 * ================================================================ */

struct logring_stats {
	float avg;
	float min;
	float max;
	float stdev;
	float trend;
};

/*
 * Compute all log_history statistics in one function: avg, min, max,
 * stdev (single-pass via E[x²]-E[x]²), and trend (older vs recent
 * half).  The trend loop accesses the same 8 elements already cached
 * by the stats loop, so fusing avoids a separate function call.
 */
static void logring_compute(const struct nap_cpu_data *d,
			    struct logring_stats *s)
{
	int i, n = d->hist_count;
	float sum, sq_sum, val, avg, inv_n, diff;

	if (n == 0) {
		*s = (struct logring_stats){ 0 };
		return;
	}

	sum = d->log_history[0];
	sq_sum = sum * sum;
	s->min = sum;
	s->max = sum;

	for (i = 1; i < n; i++) {
		val = d->log_history[i];
		sum += val;
		sq_sum += val * val;
		s->min = float_min(s->min, val);
		s->max = float_max(s->max, val);
	}

	inv_n = 1.0f / (float)n;
	avg = sum * inv_n;
	s->avg = avg;

	if (n < 2) {
		s->stdev = 0.0f;
		s->trend = 0.0f;
		return;
	}

	/* Var = E[x²] - E[x]² — single-pass, no second loop needed */
	diff = sq_sum * inv_n - avg * avg;
	s->stdev = __builtin_sqrtf(float_max(diff, 0.0f));

	/* Trend: compare older half vs recent half of the ring buffer */
	if (n < NAP_HISTORY_SIZE || avg == 0.0f) {
		s->trend = 0.0f;
	} else {
		int base = d->hist_idx;
		float older = 0.0f, recent = 0.0f;

		for (i = 0; i < 4; i++) {
			older  += d->log_history[(base + i) & 7];
			recent += d->log_history[(base + 4 + i) & 7];
		}
		s->trend = (recent - older) / (4.0f * avg);
	}
}

static u64 compute_irq_rate(const struct nap_cpu_data *d, u64 elapsed_ns)
{
	u64 cur_irq = kstat_cpu_irqs_sum(smp_processor_id());
	u64 delta_irq = cur_irq - d->prev_irq_count;

	if (elapsed_ns == 0)
		return 0;
	return div_u64(delta_irq * NSEC_PER_SEC, elapsed_ns);
}

static void nap_extract_features(struct cpuidle_driver *drv,
				 struct cpuidle_device *dev,
				 float out[NAP_NUM_FEATURES],
				 s64 latency_req)
{
	struct nap_cpu_data *d = this_cpu_ptr(&nap_data);
	struct logring_stats lr;
	ktime_t sleep_length, delta_tick;
	u64 busy_ns;
	float log_inputs[4] __attribute__((aligned(16)));
	float log_results[4] __attribute__((aligned(16)));

	sleep_length = tick_nohz_get_sleep_length(&delta_tick);

	busy_ns = local_clock() - d->prev_idle_exit;

	log_inputs[0] = (float)ktime_to_ns(sleep_length);
	log_inputs[1] = (float)dev->last_residency_ns;
	log_inputs[2] = (float)busy_ns;
	log_inputs[3] = (float)compute_irq_rate(d, busy_ns);

	log_inputs[0] = float_max(log_inputs[0], 1.0f);
	log_inputs[1] = float_max(log_inputs[1], 1.0f);
	log_inputs[2] = float_max(log_inputs[2], 1.0f);
	log_inputs[3] = float_max(log_inputs[3], 1.0f);

	{
		v4sf log_in  = *(const v4sf *)log_inputs;
		v4sf log_out = fast_log2f_sse(log_in);
		*(v4sf *)log_results = log_out;
	}

	{
		int prev = (d->hist_idx - 1 + NAP_HISTORY_SIZE) % NAP_HISTORY_SIZE;
		d->log_history[prev] = log_results[1];
	}

	/* Compute log_history statistics in a single pass */
	logring_compute(d, &lr);

	/* Group A: Time prediction */
	out[0] = log_results[0];
	out[1] = log_results[1];
	out[2] = lr.avg;
	out[3] = lr.stdev;

	/* Group B: Pattern analysis */
	out[4] = lr.min;
	out[5] = lr.max;
	out[6] = lr.trend;

	/* Sign-preserving log2 scale: maps ±N µs → ±log2(N+1) */
	{
		float err_f = (float)(d->last_prediction_error / 1000);

		out[10] = (err_f >= 0.0f)
			?  fast_log2f(err_f + 1.0f)
			: -fast_log2f(-err_f + 1.0f);
	}
	out[11] = log_results[2];

	/* Group D: External signals */
	out[12] = tick_nohz_tick_stopped() ? 1.0f : 0.0f;
	out[13] = (float)min_t(unsigned int, nr_iowait_cpu(dev->cpu), 8)
		  * 0.125f;
	out[15] = log_results[3];

	/*
	 * Batch 4 ratio features using SSE rcpps (~12-bit reciprocal
	 * approximation) + mulps.  Replaces 4× divss (~44-56 cycles)
	 * with 1× rcpps + 1× mulps (~5-8 cycles).  The 12-bit precision
	 * is adequate for NN inputs — weights compensate for any bias.
	 *
	 * Denominators are set to 1.0f when zero to avoid rcpps(0)=Inf
	 * and the resulting NaN from 0*Inf.
	 */
	{
		float nums[4] __aligned(16);
		float dens[4] __aligned(16);
		v4sf ratios;
		bool lat_valid;

		/* out[7]: short_count / total_count */
		nums[0] = (float)d->short_count;
		dens[0] = (d->total_count > 0)
			? (float)d->total_count : 1.0f;

		/* out[8]: total_above / total_usage */
		nums[1] = (float)d->total_above;
		dens[1] = (d->total_usage > 0)
			? (float)d->total_usage : 1.0f;

		/* out[9]: intercept_recent / intercept_window */
		nums[2] = (float)d->intercept_recent;
		dens[2] = (d->intercept_window > 0)
			? (float)d->intercept_window : 1.0f;

		/* out[14]: latency_req / deepest_lat */
		{
			u64 deepest_lat =
				drv->states[drv->state_count - 1]
				    .exit_latency_ns;

			lat_valid = (latency_req < S64_MAX &&
				     deepest_lat > 0);
			nums[3] = lat_valid ? (float)latency_req : 1.0f;
			dens[3] = lat_valid ? (float)deepest_lat : 1.0f;
		}

		ratios = *(const v4sf *)nums
			 * __builtin_ia32_rcpps(*(const v4sf *)dens);
		out[7]  = ratios[0];
		out[8]  = ratios[1];
		out[9]  = ratios[2];
		out[14] = ratios[3];
	}

	d->last_predicted_ns = ktime_to_ns(sleep_length);
}

/* ================================================================
 * Selection helpers
 * ================================================================ */

static int compute_ideal_state(struct cpuidle_driver *drv, u64 actual_ns)
{
	int i, best = 0;
	s64 best_score = 0;

	for (i = 1; i < drv->state_count; i++) {
		struct cpuidle_state *s = &drv->states[i];
		s64 score;

		/*
		 * Responsiveness criterion: only consider a state "ideal" if
		 * the CPU slept long enough to cover both the minimum residency
		 * and the exit latency.  This ensures the wakeup did not incur
		 * avoidable latency that the caller would have felt.
		 */
		if (actual_ns < s->target_residency_ns + s->exit_latency_ns)
			continue;

		/*
		 * Score: net sleep benefit with exit latency as a fair cost.
		 * The former (i+1) depth multiplier is removed; it biased the
		 * label toward deeper states regardless of actual benefit and
		 * worked against responsiveness.
		 */
		score = (s64)(actual_ns - s->target_residency_ns)
			- (s64)s->exit_latency_ns;

		if (score > best_score) {
			best = i;
			best_score = score;
		}
	}
	return best;
}

static int nap_nn_argmax(const float *output, int n)
{
	int i, best = 0;

	for (i = 1; i < n; i++)
		if (output[i] > output[best])
			best = i;
	return best;
}

/* ================================================================
 * FPU entry point for nap_select
 *
 * Called within kernel_fpu_begin()/kernel_fpu_end().
 * Returns: selected idle state index (>= 0), or -1 to fall back
 *          to the integer heuristic.
 * ================================================================ */

int nap_fpu_select(struct cpuidle_driver *drv,
		   struct cpuidle_device *dev,
		   struct nap_cpu_data *d)
{
	s64 latency_req = cpuidle_governor_latency_req(dev->cpu);

	/* Handle deferred weight reset (set by sysfs or nap_enable) */
	if (unlikely(d->reset_pending)) {
		nap_init_weights(&d->weights, drv);
		d->converged = false;
		d->ema_accuracy = 0;
		d->stats.learn_count = 0;
		d->needs_learn = false;
		d->reset_pending = false;
	}

	/* Deferred learning (always, even during warmup) */
	if (d->needs_learn) {
		int ideal = compute_ideal_state(drv, d->learn_actual_ns);
		int nn_best = nap_nn_argmax(d->nn_output, drv->state_count);
		bool hit = (nn_best == ideal);

		d->stats.learn_count++;

		/* Track NN accuracy for convergence detection */
		if (!d->converged) {
			d->ema_accuracy += (hit ? 64 : 0)
					   - (d->ema_accuracy >> 4);

			if (d->stats.learn_count >= d->warmup_threshold &&
			    d->ema_accuracy >= d->convergence_thresh) {
				d->converged = true;
				pr_info("nap: cpu%d converged (ema=%u/%u, learns=%llu)\n",
					smp_processor_id(),
					d->ema_accuracy,
					d->convergence_thresh,
					d->stats.learn_count);
			}
		}

		if (!hit)
			nap_nn_learn(d, ideal);
		d->needs_learn = false;
	}

	/*
	 * Feature extraction + NN forward pass.
	 * Write directly to per-CPU arrays — no intermediate buffers.
	 * features_f32 and nn_output are __aligned(64) in nap_cpu_data,
	 * satisfying AVX-512 vmovaps requirements.
	 */
	nap_extract_features(drv, dev, d->features_f32, latency_req);
	nap_nn_forward(d->features_f32, d->nn_output, d->hidden_out,
		       &d->weights);

	if (unlikely(!d->converged))
		return -1; /* Caller uses heuristic */

	/* NN-based selection */
	{
		float best_score = d->nn_output[0];
		int idx = 0, i;

		for (i = 1; i < drv->state_count; i++) {
			if (dev->states_usage[i].disable)
				continue;
			if (drv->states[i].exit_latency_ns > latency_req)
				continue;
			if (d->nn_output[i] > best_score) {
				best_score = d->nn_output[i];
				idx = i;
			}
		}
		return idx;
	}
}
