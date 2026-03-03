// SPDX-License-Identifier: GPL-2.0
/*
 * nap_nn_avx512.c — AVX-512F forward pass and backpropagation for the nap MLP
 *
 * Uses 512-bit zmm registers. Hidden layer fits in a single zmm (16 floats).
 * Forward pass uses 4-way accumulator splitting to break FMA dependency chains.
 *
 * Must be called within kernel_fpu_begin/end.
 * Compiled with: CFLAGS += -mavx512f
 */

#include "nap.h"

/*
 * Inline asm wrappers for AVX-512F operations.
 *
 * Eliminates all #ifdef __clang__ / GCC builtin differences.
 * "v" constraint = xmm/ymm/zmm (AVX-512 register class).
 * The assembler selects EVEX encoding automatically for zmm operands.
 */
static inline v16sf v16sf_max(v16sf a, v16sf b)
{
	asm("vmaxps %2, %1, %0" : "=v"(a) : "v"(a), "vm"(b));
	return a;
}

static inline v16sf v16sf_min(v16sf a, v16sf b)
{
	asm("vminps %2, %1, %0" : "=v"(a) : "v"(a), "vm"(b));
	return a;
}

/* FMA: a*b+c — vfmadd231ps with c as read-write destination,
 * matching the accumulator pattern used in forward/learn passes. */
static inline v16sf v16sf_fmadd(v16sf a, v16sf b, v16sf c)
{
	/* vfmadd231ps: dest = src1 * src2 + dest → c = a*b + c */
	asm("vfmadd231ps %2, %1, %0" : "+v"(c) : "v"(a), "vm"(b));
	return c;
}

/*
 * Split forward pass into noinline helpers so that each layer gets its
 * own stack frame.  Without this, the compiler allocates spill slots for
 * both layers simultaneously, pushing the combined frame over 2 KiB.
 */
static noinline void
fwd_avx512_hidden(const float *input, float *hidden_save,
		  const struct nap_weights *w)
{
	int j;

	/*
	 * === Hidden layer: 16 outputs = 1×zmm, 4-way accumulator ===
	 *
	 * FMA has 4-cycle latency but 0.5-cycle throughput (2 ports).
	 * A single accumulator chain: 16 × 4 = 64 cycles (latency-bound).
	 * Four independent chains: max(4×4=16cy, throughput=16cy) ≈ 16-20cy.
	 * The compiler cannot do this automatically (-O2, no -ffast-math).
	 */
	v16sf a0 = *(const v16sf *)&w->b_h1[0];
	v16sf a1 = V16SF_SET1(0.0f);
	v16sf a2 = V16SF_SET1(0.0f);
	v16sf a3 = V16SF_SET1(0.0f);

	for (j = 0; j < NAP_INPUT_SIZE; j += 4) {
		a0 = v16sf_fmadd(*(const v16sf *)&w->w_h1[j][0],
				  V16SF_SET1(input[j]), a0);
		a1 = v16sf_fmadd(*(const v16sf *)&w->w_h1[j + 1][0],
				  V16SF_SET1(input[j + 1]), a1);
		a2 = v16sf_fmadd(*(const v16sf *)&w->w_h1[j + 2][0],
				  V16SF_SET1(input[j + 2]), a2);
		a3 = v16sf_fmadd(*(const v16sf *)&w->w_h1[j + 3][0],
				  V16SF_SET1(input[j + 3]), a3);
	}

	/* ReLU: max(0, x) */
	{
		v16sf zero = V16SF_SET1(0.0f);
		v16sf acc = (a0 + a1) + (a2 + a3);

		acc = v16sf_max(acc, zero);
		*(v16sf *)&hidden_save[0] = acc;
	}
}

static noinline void
fwd_avx512_output(float *output, const float *hidden_save,
		  const struct nap_weights *w)
{
	int j;

	/* === Output layer: 1×zmm, 4-way accumulator === */
	v16sf a0 = *(const v16sf *)&w->b_out[0];
	v16sf a1 = V16SF_SET1(0.0f);
	v16sf a2 = V16SF_SET1(0.0f);
	v16sf a3 = V16SF_SET1(0.0f);

	for (j = 0; j < NAP_HIDDEN_SIZE; j += 4) {
		a0 = v16sf_fmadd(*(const v16sf *)&w->w_out[j][0],
				  V16SF_SET1(hidden_save[j]), a0);
		a1 = v16sf_fmadd(*(const v16sf *)&w->w_out[j + 1][0],
				  V16SF_SET1(hidden_save[j + 1]), a1);
		a2 = v16sf_fmadd(*(const v16sf *)&w->w_out[j + 2][0],
				  V16SF_SET1(hidden_save[j + 2]), a2);
		a3 = v16sf_fmadd(*(const v16sf *)&w->w_out[j + 3][0],
				  V16SF_SET1(hidden_save[j + 3]), a3);
	}

	*(v16sf *)&output[0] = (a0 + a1) + (a2 + a3);
}

void nap_nn_forward_avx512(const float *input,
			   float *output,
			   float *hidden_save,
			   const struct nap_weights *w)
{
	fwd_avx512_hidden(input, hidden_save, w);
	fwd_avx512_output(output, hidden_save, w);
}

/* zmm clamp: max(min(v, hi), lo) */
static inline v16sf v16sf_clamp(v16sf v, v16sf lo, v16sf hi)
{
	return v16sf_max(v16sf_min(v, hi), lo);
}

/*
 * Online learning (backpropagation) — AVX-512F
 *
 * Split into three noinline phases so each gets its own stack frame,
 * preventing zmm spill slots from accumulating into a single frame.
 * Scratch buffers live on the caller's stack and are passed by pointer.
 *
 * Output layer (10 neurons): 1×zmm + 2×ymm hsum, with FMA
 * Hidden layer (16 neurons): 1×zmm, with FMA
 */

/* Phase 1: output error, output weight/bias update (zmm) */
static noinline void
learn_avx512_output(struct nap_cpu_data *d, int ideal, float *d_out)
{
	int j;
	float lr = (float)d->learning_rate_millths / 1000.0f;
	float clamp_val = (float)d->max_grad_norm_millths / 1000.0f;
	v16sf v16_neg_lr = V16SF_SET1(-lr);
	v16sf v16_cl_hi  = V16SF_SET1(clamp_val);
	v16sf v16_cl_lo  = V16SF_SET1(-clamp_val);
	v16sf vd;

	/* Output error: nn_output - one_hot(ideal) */
	__builtin_memcpy(d_out, d->nn_output, NAP_OUTPUT_STRIDE * sizeof(float));
	d_out[ideal] -= 1.0f;

	vd = *(const v16sf *)&d_out[0];

	/* Output weight update: w_out[j][i] -= lr * clamp(h[j] * d_out[i]) */
	for (j = 0; j < NAP_HIDDEN_SIZE; j++) {
		v16sf vh = V16SF_SET1(d->hidden_out[j]);
		v16sf *w = (v16sf *)&d->weights.w_out[j][0];

		*w = v16sf_fmadd(v16_neg_lr,
				 v16sf_clamp(vh * vd, v16_cl_lo, v16_cl_hi),
				 *w);
	}

	/* Output bias update */
	{
		v16sf *b = (v16sf *)&d->weights.b_out[0];

		*b = v16sf_fmadd(v16_neg_lr,
				 v16sf_clamp(vd, v16_cl_lo, v16_cl_hi), *b);
	}
}

/* Phase 2: hidden gradient via 2×ymm horizontal sum */
static noinline void
learn_avx512_hidden_grad(struct nap_cpu_data *d,
			 const float *d_out, float *d_hid)
{
	int j;
	v8sf vd_lo = *(const v8sf *)&d_out[0];
	v8sf vd_hi = *(const v8sf *)&d_out[8];

	for (j = 0; j < NAP_HIDDEN_SIZE; j++) {
		const v8sf *w = (const v8sf *)&d->weights.w_out[j][0];
		v8sf s = w[0] * vd_lo + w[1] * vd_hi;
		v4sf lo = __builtin_ia32_vextractf128_ps256(s, 0);
		v4sf hi = __builtin_ia32_vextractf128_ps256(s, 1);
		v4sf s4 = lo + hi;
		float sum = s4[0] + s4[1] + s4[2] + s4[3];

		d_hid[j] = (d->hidden_out[j] > 0) ? sum : 0;
	}
}

/* Phase 3: hidden weight/bias update (zmm) */
static noinline void
learn_avx512_hidden_update(struct nap_cpu_data *d, const float *d_hid)
{
	int i;
	float lr = (float)d->learning_rate_millths / 1000.0f;
	float clamp_val = (float)d->max_grad_norm_millths / 1000.0f;
	v16sf v16_neg_lr = V16SF_SET1(-lr);
	v16sf v16_cl_hi  = V16SF_SET1(clamp_val);
	v16sf v16_cl_lo  = V16SF_SET1(-clamp_val);
	v16sf dh = *(const v16sf *)&d_hid[0];

	/* Hidden weight update: w_h1[i][j] -= lr * clamp(feat[i] * d_hid[j])
	 * 16 hidden neurons = 1×zmm per input row */
	for (i = 0; i < NAP_INPUT_SIZE; i++) {
		v16sf vf = V16SF_SET1(d->features_f32[i]);
		v16sf *w = (v16sf *)&d->weights.w_h1[i][0];

		*w = v16sf_fmadd(v16_neg_lr,
				 v16sf_clamp(vf * dh,
					     v16_cl_lo, v16_cl_hi),
				 *w);
	}

	/* Hidden bias update */
	{
		v16sf *b = (v16sf *)&d->weights.b_h1[0];

		*b = v16sf_fmadd(v16_neg_lr,
				 v16sf_clamp(dh, v16_cl_lo, v16_cl_hi),
				 *b);
	}
}

void nap_nn_learn_avx512(struct nap_cpu_data *d, int ideal)
{
	float d_out[NAP_OUTPUT_STRIDE] __aligned(64);
	float d_hid[NAP_HIDDEN_SIZE] __aligned(64);

	learn_avx512_output(d, ideal, d_out);
	learn_avx512_hidden_grad(d, d_out, d_hid);
	learn_avx512_hidden_update(d, d_hid);
}
