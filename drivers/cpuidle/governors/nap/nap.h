/* SPDX-License-Identifier: GPL-2.0 */
#ifndef NAP_H
#define NAP_H

#include <linux/cpuidle.h>
#include <linux/jump_label.h>
#include <linux/ktime.h>

/* ================================================================
 * Neural network dimensions
 * ================================================================ */

#define NAP_INPUT_SIZE    16
#define NAP_HIDDEN_SIZE   16
#define NAP_OUTPUT_SIZE   CPUIDLE_STATE_MAX
#define NAP_OUTPUT_STRIDE 16  /* SIMD-padded output dimension */

static_assert(NAP_OUTPUT_SIZE <= NAP_OUTPUT_STRIDE,
	      "NAP_OUTPUT_SIZE must not exceed NAP_OUTPUT_STRIDE");

/*
 * Neural network weight structure for a 16→16→10 MLP.
 *
 * Column-major storage: w_h1[j][i] = weight from input j to hidden neuron i.
 * This layout enables efficient column-wise matrix-vector products where
 * each input broadcasts across all output neurons via SIMD FMA.
 *
 * Output arrays use NAP_OUTPUT_STRIDE (16) instead of NAP_OUTPUT_SIZE (10)
 * so that each row is 64-byte aligned and fits exactly in SSE (4×xmm),
 * AVX2 (2×ymm), or AVX-512 (1×zmm) registers with no scalar tail.
 *
 * __aligned(64) ensures both AVX2 (vmovaps, 32-byte) and AVX-512
 * (vmovaps, 64-byte) aligned loads work correctly.
 */
struct nap_weights {
	/* Hidden layer: input[16] → hidden[16] */
	float w_h1[NAP_INPUT_SIZE][NAP_HIDDEN_SIZE];      /* 256 params */
	float b_h1[NAP_HIDDEN_SIZE];                       /* 16 params  */
	/* Output layer: hidden[16] → output[10], padded to 16 */
	float w_out[NAP_HIDDEN_SIZE][NAP_OUTPUT_STRIDE];   /* 160+pad    */
	float b_out[NAP_OUTPUT_STRIDE];                    /* 10+pad     */
} __aligned(64);

/* ISA-specific forward pass implementations */
void nap_nn_forward_sse2(const float *input, float *output,
			 float *hidden_save, const struct nap_weights *w);
void nap_nn_forward_avx2(const float *input, float *output,
			 float *hidden_save, const struct nap_weights *w);
void nap_nn_forward_avx512(const float *input, float *output,
			   float *hidden_save, const struct nap_weights *w);

/* ISA-specific online learning (backpropagation) */
struct nap_cpu_data;
void nap_nn_learn_sse2(struct nap_cpu_data *d, int ideal);
void nap_nn_learn_avx2(struct nap_cpu_data *d, int ideal);
void nap_nn_learn_avx512(struct nap_cpu_data *d, int ideal);

/* Static keys for ISA dispatch (defined in nap.c) */
DECLARE_STATIC_KEY_FALSE(nap_use_avx512);
DECLARE_STATIC_KEY_FALSE(nap_use_avx2);

/* ================================================================
 * SIMD type definitions and helpers (GCC vector extensions)
 *
 * Only available when compiled with FPU/SSE flags (nap_fpu.c,
 * nap_nn_*.c).  nap.c is compiled without FPU flags and must
 * not see these definitions.
 *
 * <immintrin.h> is a userspace header and cannot be used in kernel.
 * We use __attribute__((__vector_size__())) and __builtin_ia32_*.
 * ================================================================ */

#ifdef __SSE2__

typedef float v4sf  __attribute__((__vector_size__(16)));   /* xmm: 4×float  */
typedef int   v4si  __attribute__((__vector_size__(16)));   /* xmm: 4×int32  */
typedef float v8sf  __attribute__((__vector_size__(32)));   /* ymm: 8×float  */
typedef float v16sf __attribute__((__vector_size__(64)));   /* zmm: 16×float */

/* Broadcast helpers */
#define V4SF_SET1(x)  ((v4sf){ (x), (x), (x), (x) })
#define V4SI_SET1(x)  ((v4si){ (x), (x), (x), (x) })
#define V8SF_SET1(x)  ((v8sf){ (x),(x),(x),(x),(x),(x),(x),(x) })
#define V8SF_ZERO     V8SF_SET1(0.0f)
#define V16SF_SET1(x) ((v16sf){ (x),(x),(x),(x),(x),(x),(x),(x), \
                                (x),(x),(x),(x),(x),(x),(x),(x) })

/* Unaligned load/store helpers */
static inline v4sf v4sf_loadu(const float *p)
{
	v4sf result;
	__builtin_memcpy(&result, p, sizeof(result));
	return result;
}

static inline void v4sf_storeu(float *p, v4sf v)
{
	__builtin_memcpy(p, &v, sizeof(v));
}

#ifdef __AVX__
static inline v8sf v8sf_loadu(const float *p)
{
	v8sf result;
	__builtin_memcpy(&result, p, sizeof(result));
	return result;
}

static inline void v8sf_storeu(float *p, v8sf v)
{
	__builtin_memcpy(p, &v, sizeof(v));
}
#endif /* __AVX__ */

/* Scalar/vector clamp helpers */
static inline float fclampf(float v, float lo, float hi)
{
	if (v < lo) return lo;
	if (v > hi) return hi;
	return v;
}

static inline v4sf v4sf_clamp(v4sf v, v4sf lo, v4sf hi)
{
	return __builtin_ia32_maxps(__builtin_ia32_minps(v, hi), lo);
}

/* Type punning: float ↔ int reinterpret (no instruction generated) */
static inline v4si v4sf_as_v4si(v4sf v)
{
	union { v4sf f; v4si i; } u = { .f = v };
	return u.i;
}

static inline v4sf v4si_as_v4sf(v4si v)
{
	union { v4si i; v4sf f; } u = { .i = v };
	return u.f;
}

/*
 * fast_log2f_sse() — Compute log2 of 4 floats simultaneously using SSE2
 *
 * Cost: ~15 cycles for 4 values (~4 cycles per value)
 */
static inline v4sf fast_log2f_sse(v4sf x)
{
	const v4si mask_exp  = V4SI_SET1(0xFF);
	const v4si bias      = V4SI_SET1(127);
	const v4si mask_mant = V4SI_SET1(0x7FFFFF);
	const v4si exp_bias  = V4SI_SET1(127 << 23);

	v4si xi    = v4sf_as_v4si(x);
	v4si exp_i = (xi >> 23) & mask_exp;
	exp_i      = exp_i - bias;
	v4sf e     = __builtin_convertvector(exp_i, v4sf);

	v4si mant_i = (xi & mask_mant) | exp_bias;
	v4sf m      = v4si_as_v4sf(mant_i) - V4SF_SET1(1.0f);

	v4sf p;
	p = m * V4SF_SET1(0.4808f);
	p = V4SF_SET1(0.7213f) - p;
	p = m * p;
	p = V4SF_SET1(1.4425f) - p;
	p = m * p;

	return e + p;
}

#endif /* __SSE2__ */

/* ================================================================
 * Feature extraction
 * ================================================================ */

#define NAP_HISTORY_SIZE     8
#define NAP_NUM_FEATURES     16
#define NAP_SHORT_THRESH_NS  (100 * NSEC_PER_USEC)

struct nap_stats {
	u64 total_selects;
	u64 total_residency_ns;
	u64 undershoot_count;
	u64 learn_count;
};

struct nap_cpu_data {
	/* Ring buffer */
	u64   history[NAP_HISTORY_SIZE];
	float log_history[NAP_HISTORY_SIZE];
	int   hist_idx;
	int   hist_count;

	/* Statistics tracking */
	u64   total_above;
	u64   total_usage;
	u64   intercept_recent;
	u64   intercept_window;
	u64   short_count;
	u64   total_count;

	/* External signal tracking */
	u64     prev_irq_count;
	u64     prev_idle_exit;
	s64     last_predicted_ns;
	s64     last_prediction_error;

	/* select/reflect handoff */
	int   last_selected_idx;
	/*
	 * nn_output[], hidden_out[], features_f32[] are written with
	 * aligned SIMD stores in nap_nn_forward_{sse2,avx2,avx512}()
	 * and nap_extract_features():
	 *   SSE2:   movaps  (16-byte aligned)
	 *   AVX2:   vmovaps (32-byte aligned)
	 *   AVX-512: vmovaps zmm (64-byte aligned)
	 * Without __aligned(64), the natural struct offset would be
	 * only 4-byte aligned, causing #GP faults in the idle task.
	 *
	 * nap_fpu_select() writes directly to these arrays, avoiding
	 * intermediate buffers and memcpy.
	 */
	float nn_output[NAP_OUTPUT_STRIDE] __aligned(64);
	float hidden_out[NAP_HIDDEN_SIZE] __aligned(64);
	float features_f32[NAP_NUM_FEATURES] __aligned(64);

	/* Backprop scratch (avoids 64-byte-aligned stack arrays) */
	float learn_d_out[NAP_OUTPUT_STRIDE] __aligned(64);
	float learn_d_hid[NAP_HIDDEN_SIZE] __aligned(64);

	/* Deferred learning data */
	bool  needs_learn;
	u64   learn_actual_ns;

	/* Online learning */
	struct nap_weights weights;
	bool  learning_mode;
	unsigned int learning_rate_millths;
	unsigned int max_grad_norm_millths;
	int   learn_interval;
	int   learn_counter;
	unsigned int warmup_threshold;	/* min learns before convergence check */
	unsigned int convergence_thresh;	/* EMA accuracy threshold (x1024) */
	unsigned int ema_accuracy;	/* EMA of NN hit rate (x1024 = 100%) */
	bool converged;			/* true after NN accuracy converges */
	bool reset_pending;		/* set by sysfs, consumed by nap_select */

	/* sysfs statistics */
	struct nap_stats stats;
};

DECLARE_PER_CPU(struct nap_cpu_data, nap_data);

/* FPU entry point (nap_fpu.c) — call only within kernel_fpu_begin/end */
int nap_fpu_select(struct cpuidle_driver *drv,
		   struct cpuidle_device *dev,
		   struct nap_cpu_data *d);

/* sysfs interface */
int  nap_sysfs_init(void);
void nap_sysfs_exit(void);

#endif /* NAP_H */
