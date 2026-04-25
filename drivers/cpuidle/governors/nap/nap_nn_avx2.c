// SPDX-License-Identifier: GPL-2.0
/*
 * nap_nn_avx2.c — AVX2+FMA forward pass and backpropagation for the nap MLP
 *
 * Uses 256-bit ymm registers and vfmadd231ps for fused multiply-add.
 *
 * Must be called within kernel_fpu_begin/end.
 * Compiled with: CFLAGS += -mavx2 -mfma
 */

#include "nap.h"

/* Aligned load/store: GCC translates v8sf* dereference to vmovaps */
static inline v8sf v8sf_load(const float *p)   { return *(const v8sf *)p; }
static inline void v8sf_store(float *p, v8sf v) { *(v8sf *)p = v; }

/* FMA: a*b+c — inline asm guarantees a single vfmaddps instruction
 * on both GCC and Clang without relying on compiler-specific builtins
 * or -ffp-contract settings.
 *
 * vfmadd231ps chosen over 213: the accumulator/destination is c ("+x"),
 * so live-through registers like v_neg_lr in the learn pass stay in
 * read-only operands and need no extra vmovaps copy. */
static inline v8sf v8sf_fmadd(v8sf a, v8sf b, v8sf c)
{
	/* vfmadd231ps: dest = src1 * src2 + dest → c = a*b + c */
	asm("vfmadd231ps %2, %1, %0" : "+x"(c) : "x"(a), "xm"(b));
	return c;
}

/*
 * Split forward pass into noinline helpers so that each layer gets its
 * own stack frame.  Without this, 8+8 ymm accumulators push the combined
 * frame over 2 KiB due to register spills.
 */
static noinline void
fwd_avx2_hidden(const float *input, float *hidden_save,
		const struct nap_weights *w)
{
	int j;

	/*
	 * === Hidden layer: 16 outputs = 2×ymm, 4-way accumulator ===
	 *
	 * Break the FMA dependency chain (16 × 4cy = 64cy latency-bound)
	 * into 4 independent chains (4 × 4cy = 16cy each, overlapped).
	 * 8 ymm accumulators: a{0..3}_lo (lanes 0-7), a{0..3}_hi (lanes 8-15).
	 */
	v8sf a0_lo = v8sf_load(&w->b_h1[0]);
	v8sf a0_hi = v8sf_load(&w->b_h1[8]);
	v8sf a1_lo = V8SF_ZERO, a1_hi = V8SF_ZERO;
	v8sf a2_lo = V8SF_ZERO, a2_hi = V8SF_ZERO;
	v8sf a3_lo = V8SF_ZERO, a3_hi = V8SF_ZERO;

	for (j = 0; j < NAP_INPUT_SIZE; j += 4) {
		v8sf x0 = V8SF_SET1(input[j]);
		v8sf x1 = V8SF_SET1(input[j + 1]);
		v8sf x2 = V8SF_SET1(input[j + 2]);
		v8sf x3 = V8SF_SET1(input[j + 3]);

		a0_lo = v8sf_fmadd(v8sf_load(&w->w_h1[j][0]), x0, a0_lo);
		a0_hi = v8sf_fmadd(v8sf_load(&w->w_h1[j][8]), x0, a0_hi);
		a1_lo = v8sf_fmadd(v8sf_load(&w->w_h1[j + 1][0]), x1, a1_lo);
		a1_hi = v8sf_fmadd(v8sf_load(&w->w_h1[j + 1][8]), x1, a1_hi);
		a2_lo = v8sf_fmadd(v8sf_load(&w->w_h1[j + 2][0]), x2, a2_lo);
		a2_hi = v8sf_fmadd(v8sf_load(&w->w_h1[j + 2][8]), x2, a2_hi);
		a3_lo = v8sf_fmadd(v8sf_load(&w->w_h1[j + 3][0]), x3, a3_lo);
		a3_hi = v8sf_fmadd(v8sf_load(&w->w_h1[j + 3][8]), x3, a3_hi);
	}

	/* ReLU: max(0, x) */
	{
		v8sf zero = V8SF_ZERO;
		v8sf lo = (a0_lo + a1_lo) + (a2_lo + a3_lo);
		v8sf hi = (a0_hi + a1_hi) + (a2_hi + a3_hi);

		lo = __builtin_ia32_maxps256(lo, zero);
		hi = __builtin_ia32_maxps256(hi, zero);
		v8sf_store(&hidden_save[0], lo);
		v8sf_store(&hidden_save[8], hi);
	}
}

static noinline void
fwd_avx2_output(float *output, const float *hidden_save,
		const struct nap_weights *w)
{
	int j;

	/* === Output layer: 2×ymm, 4-way accumulator === */
	v8sf a0_lo = v8sf_load(&w->b_out[0]);
	v8sf a0_hi = v8sf_load(&w->b_out[8]);
	v8sf a1_lo = V8SF_ZERO, a1_hi = V8SF_ZERO;
	v8sf a2_lo = V8SF_ZERO, a2_hi = V8SF_ZERO;
	v8sf a3_lo = V8SF_ZERO, a3_hi = V8SF_ZERO;

	for (j = 0; j < NAP_HIDDEN_SIZE; j += 4) {
		v8sf h0 = V8SF_SET1(hidden_save[j]);
		v8sf h1 = V8SF_SET1(hidden_save[j + 1]);
		v8sf h2 = V8SF_SET1(hidden_save[j + 2]);
		v8sf h3 = V8SF_SET1(hidden_save[j + 3]);

		a0_lo = v8sf_fmadd(v8sf_load(&w->w_out[j][0]), h0, a0_lo);
		a0_hi = v8sf_fmadd(v8sf_load(&w->w_out[j][8]), h0, a0_hi);
		a1_lo = v8sf_fmadd(v8sf_load(&w->w_out[j + 1][0]), h1, a1_lo);
		a1_hi = v8sf_fmadd(v8sf_load(&w->w_out[j + 1][8]), h1, a1_hi);
		a2_lo = v8sf_fmadd(v8sf_load(&w->w_out[j + 2][0]), h2, a2_lo);
		a2_hi = v8sf_fmadd(v8sf_load(&w->w_out[j + 2][8]), h2, a2_hi);
		a3_lo = v8sf_fmadd(v8sf_load(&w->w_out[j + 3][0]), h3, a3_lo);
		a3_hi = v8sf_fmadd(v8sf_load(&w->w_out[j + 3][8]), h3, a3_hi);
	}

	v8sf_store(&output[0], (a0_lo + a1_lo) + (a2_lo + a3_lo));
	v8sf_store(&output[8], (a0_hi + a1_hi) + (a2_hi + a3_hi));
}

void nap_nn_forward_avx2(const float *input,
			 float *output,
			 float *hidden_save,
			 const struct nap_weights *w)
{
	fwd_avx2_hidden(input, hidden_save, w);
	fwd_avx2_output(output, hidden_save, w);
}

/* ymm clamp: max(min(v, hi), lo) */
static inline v8sf v8sf_clamp(v8sf v, v8sf lo, v8sf hi)
{
	return __builtin_ia32_maxps256(__builtin_ia32_minps256(v, hi), lo);
}

/*
 * Online learning (backpropagation) — AVX2+FMA
 *
 * Output layer (10 neurons): 1×ymm + 2 scalars, with FMA
 * Hidden layer (16 neurons): 2×ymm, with FMA
 */
void nap_nn_learn_avx2(struct nap_cpu_data *d, int ideal)
{
	int i, j;
	float *d_out = d->learn_d_out;
	float *d_hid = d->learn_d_hid;
	float lr = (float)d->learning_rate_millths / 1000.0f;
	float clamp_val = (float)d->max_grad_norm_millths / 1000.0f;
	v8sf v_neg_lr = V8SF_SET1(-lr);
	v8sf v_cl_hi  = V8SF_SET1(clamp_val);
	v8sf v_cl_lo  = V8SF_SET1(-clamp_val);
	v8sf vd_lo, vd_hi;

	/* Output error: nn_output - one_hot(ideal) */
	__builtin_memcpy(d_out, d->nn_output, NAP_OUTPUT_STRIDE * sizeof(float));
	d_out[ideal] -= 1.0f;

	vd_lo = v8sf_load(&d_out[0]);
	vd_hi = v8sf_load(&d_out[8]);

	/* Output weight update: w_out[j][i] -= lr * clamp(h[j] * d_out[i]) */
	for (j = 0; j < NAP_HIDDEN_SIZE; j++) {
		v8sf vh = V8SF_SET1(d->hidden_out[j]);
		v8sf *w = (v8sf *)&d->weights.w_out[j][0];

		w[0] = v8sf_fmadd(v_neg_lr,
				   v8sf_clamp(vh * vd_lo, v_cl_lo, v_cl_hi),
				   w[0]);
		w[1] = v8sf_fmadd(v_neg_lr,
				   v8sf_clamp(vh * vd_hi, v_cl_lo, v_cl_hi),
				   w[1]);
	}

	/* Output bias update: b_out[i] -= lr * clamp(d_out[i]) */
	{
		v8sf *b = (v8sf *)&d->weights.b_out[0];

		b[0] = v8sf_fmadd(v_neg_lr,
				   v8sf_clamp(vd_lo, v_cl_lo, v_cl_hi), b[0]);
		b[1] = v8sf_fmadd(v_neg_lr,
				   v8sf_clamp(vd_hi, v_cl_lo, v_cl_hi), b[1]);
	}

	/* Hidden gradient: d_hid[j] = relu'(h[j]) * dot(w_out[j][:], d_out[:]) */
	for (j = 0; j < NAP_HIDDEN_SIZE; j++) {
		const v8sf *w = (const v8sf *)&d->weights.w_out[j][0];
		v8sf s = w[0] * vd_lo + w[1] * vd_hi;
		v4sf lo = __builtin_ia32_vextractf128_ps256(s, 0);
		v4sf hi = __builtin_ia32_vextractf128_ps256(s, 1);
		v4sf s4 = lo + hi;
		float sum = s4[0] + s4[1] + s4[2] + s4[3];

		d_hid[j] = (d->hidden_out[j] > 0) ? sum : 0;
	}

	/* Hidden weight update: w_h1[i][j] -= lr * clamp(feat[i] * d_hid[j]) */
	{
		v8sf dh0 = v8sf_load(&d_hid[0]);
		v8sf dh1 = v8sf_load(&d_hid[8]);

		for (i = 0; i < NAP_INPUT_SIZE; i++) {
			v8sf vf = V8SF_SET1(d->features_f32[i]);
			v8sf *w = (v8sf *)&d->weights.w_h1[i][0];

			w[0] = v8sf_fmadd(v_neg_lr,
					  v8sf_clamp(vf * dh0, v_cl_lo, v_cl_hi),
					  w[0]);
			w[1] = v8sf_fmadd(v_neg_lr,
					  v8sf_clamp(vf * dh1, v_cl_lo, v_cl_hi),
					  w[1]);
		}

		/* Hidden bias update */
		{
			v8sf *b = (v8sf *)&d->weights.b_h1[0];

			b[0] = v8sf_fmadd(v_neg_lr,
					  v8sf_clamp(dh0, v_cl_lo, v_cl_hi),
					  b[0]);
			b[1] = v8sf_fmadd(v_neg_lr,
					  v8sf_clamp(dh1, v_cl_lo, v_cl_hi),
					  b[1]);
		}
	}
}
