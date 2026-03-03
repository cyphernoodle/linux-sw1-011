// SPDX-License-Identifier: GPL-2.0
/*
 * nap_nn_sse2.c — SSE2 forward pass and backpropagation for the nap MLP
 *
 * Baseline implementation using SSE2, which is always available on x86_64.
 * No FMA — uses separate mul + add (2 instructions per MAC).
 *
 * Must be called within kernel_fpu_begin/end.
 * Compiled with: CFLAGS += -msse2
 */

#include "nap.h"

/* Aligned load/store */
static inline v4sf v4sf_load(const float *p)   { return *(const v4sf *)p; }
static inline void v4sf_store(float *p, v4sf v) { *(v4sf *)p = v; }

/* ReLU helper */
static inline v4sf v4sf_max(v4sf a, v4sf b)
{
	return __builtin_ia32_maxps(a, b);
}

void nap_nn_forward_sse2(const float *input,
			 float *output,
			 float *hidden_save,
			 const struct nap_weights *w)
{
	int j;

	/* === Hidden layer: 16 outputs = 4×xmm === */
	v4sf acc0 = v4sf_load(&w->b_h1[0]);
	v4sf acc1 = v4sf_load(&w->b_h1[4]);
	v4sf acc2 = v4sf_load(&w->b_h1[8]);
	v4sf acc3 = v4sf_load(&w->b_h1[12]);

	for (j = 0; j < NAP_INPUT_SIZE; j++) {
		v4sf x = V4SF_SET1(input[j]);
		acc0 += v4sf_load(&w->w_h1[j][0])  * x;
		acc1 += v4sf_load(&w->w_h1[j][4])  * x;
		acc2 += v4sf_load(&w->w_h1[j][8])  * x;
		acc3 += v4sf_load(&w->w_h1[j][12]) * x;
	}

	/* ReLU */
	v4sf zero = V4SF_SET1(0.0f);
	acc0 = v4sf_max(acc0, zero);
	acc1 = v4sf_max(acc1, zero);
	acc2 = v4sf_max(acc2, zero);
	acc3 = v4sf_max(acc3, zero);
	v4sf_store(&hidden_save[0],  acc0);
	v4sf_store(&hidden_save[4],  acc1);
	v4sf_store(&hidden_save[8],  acc2);
	v4sf_store(&hidden_save[12], acc3);

	/* Output layer: 16 outputs (10 active + 6 padding) = 4×xmm */
	v4sf out0 = v4sf_load(&w->b_out[0]);
	v4sf out1 = v4sf_load(&w->b_out[4]);
	v4sf out2 = v4sf_load(&w->b_out[8]);
	v4sf out3 = v4sf_load(&w->b_out[12]);

	for (j = 0; j < NAP_HIDDEN_SIZE; j++) {
		v4sf h = V4SF_SET1(hidden_save[j]);
		out0 += v4sf_load(&w->w_out[j][0])  * h;
		out1 += v4sf_load(&w->w_out[j][4])  * h;
		out2 += v4sf_load(&w->w_out[j][8])  * h;
		out3 += v4sf_load(&w->w_out[j][12]) * h;
	}

	v4sf_store(&output[0],  out0);
	v4sf_store(&output[4],  out1);
	v4sf_store(&output[8],  out2);
	v4sf_store(&output[12], out3);
}

/*
 * Online learning (backpropagation) — SSE2
 *
 * Output layer (10 neurons): 2×xmm + 2 scalars
 * Hidden layer (16 neurons): 4×xmm
 */
void nap_nn_learn_sse2(struct nap_cpu_data *d, int ideal)
{
	int i, j;
	float *d_out = d->learn_d_out;
	float *d_hid = d->learn_d_hid;
	float lr = (float)d->learning_rate_millths / 1000.0f;
	float clamp_val = (float)d->max_grad_norm_millths / 1000.0f;
	v4sf v_lr    = V4SF_SET1(lr);
	v4sf v_cl_hi = V4SF_SET1(clamp_val);
	v4sf v_cl_lo = V4SF_SET1(-clamp_val);
	v4sf vd0, vd1, vd2, vd3;

	/* Output error: nn_output - one_hot(ideal) */
	__builtin_memcpy(d_out, d->nn_output, NAP_OUTPUT_STRIDE * sizeof(float));
	d_out[ideal] -= 1.0f;

	vd0 = *(const v4sf *)&d_out[0];
	vd1 = *(const v4sf *)&d_out[4];
	vd2 = *(const v4sf *)&d_out[8];
	vd3 = *(const v4sf *)&d_out[12];

	/* Output weight update: w_out[j][i] -= lr * clamp(h[j] * d_out[i]) */
	for (j = 0; j < NAP_HIDDEN_SIZE; j++) {
		v4sf vh = V4SF_SET1(d->hidden_out[j]);
		v4sf *w = (v4sf *)&d->weights.w_out[j][0];

		w[0] -= v_lr * v4sf_clamp(vh * vd0, v_cl_lo, v_cl_hi);
		w[1] -= v_lr * v4sf_clamp(vh * vd1, v_cl_lo, v_cl_hi);
		w[2] -= v_lr * v4sf_clamp(vh * vd2, v_cl_lo, v_cl_hi);
		w[3] -= v_lr * v4sf_clamp(vh * vd3, v_cl_lo, v_cl_hi);
	}

	/* Output bias update: b_out[i] -= lr * clamp(d_out[i]) */
	{
		v4sf *b = (v4sf *)&d->weights.b_out[0];

		b[0] -= v_lr * v4sf_clamp(vd0, v_cl_lo, v_cl_hi);
		b[1] -= v_lr * v4sf_clamp(vd1, v_cl_lo, v_cl_hi);
		b[2] -= v_lr * v4sf_clamp(vd2, v_cl_lo, v_cl_hi);
		b[3] -= v_lr * v4sf_clamp(vd3, v_cl_lo, v_cl_hi);
	}

	/* Hidden gradient: d_hid[j] = relu'(h[j]) * dot(w_out[j][:], d_out[:]) */
	for (j = 0; j < NAP_HIDDEN_SIZE; j++) {
		const v4sf *w = (const v4sf *)&d->weights.w_out[j][0];
		v4sf s = w[0] * vd0 + w[1] * vd1 + w[2] * vd2 + w[3] * vd3;
		float sum = s[0] + s[1] + s[2] + s[3];

		d_hid[j] = (d->hidden_out[j] > 0) ? sum : 0;
	}

	/* Hidden weight update: w_h1[i][j] -= lr * clamp(feat[i] * d_hid[j]) */
	{
		v4sf dh0 = *(const v4sf *)&d_hid[0];
		v4sf dh1 = *(const v4sf *)&d_hid[4];
		v4sf dh2 = *(const v4sf *)&d_hid[8];
		v4sf dh3 = *(const v4sf *)&d_hid[12];

		for (i = 0; i < NAP_INPUT_SIZE; i++) {
			v4sf vf = V4SF_SET1(d->features_f32[i]);
			v4sf *w = (v4sf *)&d->weights.w_h1[i][0];

			w[0] -= v_lr * v4sf_clamp(vf * dh0, v_cl_lo, v_cl_hi);
			w[1] -= v_lr * v4sf_clamp(vf * dh1, v_cl_lo, v_cl_hi);
			w[2] -= v_lr * v4sf_clamp(vf * dh2, v_cl_lo, v_cl_hi);
			w[3] -= v_lr * v4sf_clamp(vf * dh3, v_cl_lo, v_cl_hi);
		}

		/* Hidden bias update: b_h1[j] -= lr * clamp(d_hid[j]) */
		{
			v4sf *b = (v4sf *)&d->weights.b_h1[0];

			b[0] -= v_lr * v4sf_clamp(dh0, v_cl_lo, v_cl_hi);
			b[1] -= v_lr * v4sf_clamp(dh1, v_cl_lo, v_cl_hi);
			b[2] -= v_lr * v4sf_clamp(dh2, v_cl_lo, v_cl_hi);
			b[3] -= v_lr * v4sf_clamp(dh3, v_cl_lo, v_cl_hi);
		}
	}
}
