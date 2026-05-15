// SPDX-License-Identifier: GPL-2.0
/*
 * nap.c — Neural Adaptive Predictor cpuidle governor
 *
 * A machine-learning-based cpuidle governor that uses a small MLP (16→16→10)
 * to predict the optimal idle state.  Weights are Xavier-initialized at boot
 * with exit-latency-aware output biases, then refined via online learning
 * (deferred backpropagation with SGD).
 *
 * IMPORTANT: This file is compiled WITHOUT FPU/SSE flags (normal kernel
 * compilation).  All floating-point and SIMD code lives in nap_fpu.c and
 * nap_nn_{sse2,avx2,avx512}.c, which are compiled with CC_FLAGS_FPU.
 * This separation ensures the compiler cannot emit SSE instructions in
 * governor callbacks (nap_select, nap_reflect, etc.), which would corrupt
 * userspace FPU register state.
 */

#include <linux/cpuidle.h>
#include <linux/cpu.h>
#include <linux/jump_label.h>
#include <linux/kernel_stat.h>
#include <linux/kobject.h>
#include <linux/math64.h>
#include <linux/percpu.h>
#include <linux/sched/clock.h>
#include <linux/sysfs.h>
#include <linux/string.h>
#include <asm/simd.h>
#include <asm/fpu/api.h>
#include <asm/processor.h>

#include "nap.h"

#include "../gov.h"

/**************************************************************
 * Version Information:
 */

#define CPUIDLE_NAP_PROGNAME "Nap CPUIdle Governor"
#define CPUIDLE_NAP_AUTHOR   "Masahito Suzuki"

#define CPUIDLE_NAP_VERSION  "0.1.9"

/* Governor defaults */
#define NAP_DEFAULT_LR_MILLTHS    1     /* 0.001 = 1 millths */
#define NAP_DEFAULT_INTERVAL      16    /* learn every 16 reflects */
#define NAP_DEFAULT_CLAMP_MILLTHS 1000  /* 1.0 = 1000 millths */
#define NAP_DEFAULT_WARMUP        64    /* min learns before convergence check */
#define NAP_DEFAULT_CONVERGE      768   /* 75% accuracy (x1024 scale) */

/* ================================================================
 * ISA dispatch via static keys (definitions only; dispatch in nap_fpu.c)
 * ================================================================ */

DEFINE_STATIC_KEY_FALSE(nap_use_avx512);
DEFINE_STATIC_KEY_FALSE(nap_use_avx2);

/*
 * Intel CPUs (Skylake through Sapphire Rapids) reduce core frequency
 * when executing 512-bit AVX instructions ("AVX-512 heavy" license).
 * Recovery takes ~670 µs, so using AVX-512 right before entering idle
 * can hurt wakeup latency on shallow C-states (C1).
 *
 * The cycle savings from AVX-512 over AVX2 are small (~2-4 ns after
 * accumulator splitting), far outweighed by the frequency penalty.
 * AMD Zen 4+ does not throttle on AVX-512, so we enable it there.
 */
static void __init nap_detect_simd(void)
{
	if (boot_cpu_has(X86_FEATURE_AVX512F) &&
	    boot_cpu_data.x86_vendor != X86_VENDOR_INTEL) {
		static_branch_enable(&nap_use_avx512);
		pr_info("nap: using AVX-512F (no frequency penalty)\n");
	} else if (boot_cpu_has(X86_FEATURE_FMA) &&
		   boot_cpu_has(X86_FEATURE_AVX2)) {
		static_branch_enable(&nap_use_avx2);
		if (boot_cpu_has(X86_FEATURE_AVX512F))
			pr_info("nap: using AVX2+FMA (AVX-512 avoided for frequency stability)\n");
		else
			pr_info("nap: using AVX2+FMA\n");
	} else {
		pr_info("nap: using SSE2\n");
	}
}

/* ================================================================
 * Per-CPU data
 * ================================================================ */

DEFINE_PER_CPU(struct nap_cpu_data, nap_data);
static struct cpuidle_driver *nap_cached_drv;

/* ================================================================
 * Reflect-time updates (integer-only, no FPU needed)
 * ================================================================ */

static void nap_history_update(struct nap_cpu_data *d, u64 measured_ns)
{
	d->history[d->hist_idx] = measured_ns;
	d->hist_idx = (d->hist_idx + 1) % NAP_HISTORY_SIZE;
	if (d->hist_count < NAP_HISTORY_SIZE)
		d->hist_count++;

	d->total_count++;
	if (measured_ns < NAP_SHORT_THRESH_NS)
		d->short_count++;
}

static void nap_update_hit_intercept(struct nap_cpu_data *d,
				     struct cpuidle_driver *drv,
				     struct cpuidle_device *dev,
				     int selected_idx, u64 measured_ns)
{
	d->total_usage++;

	if (selected_idx + 1 < drv->state_count &&
	    measured_ns > drv->states[selected_idx + 1].target_residency_ns)
		d->total_above++;

	d->intercept_window++;
	if (measured_ns < (u64)d->last_predicted_ns)
		d->intercept_recent++;

	if (d->intercept_window >= 1024) {
		d->intercept_window >>= 1;
		d->intercept_recent >>= 1;
	}
}

static void nap_update_external_signals(struct nap_cpu_data *d, int cpu)
{
	u64 cur_irq = kstat_cpu_irqs_sum(cpu);

	d->prev_irq_count = cur_irq;
	d->prev_idle_exit = local_clock();
}

/* ================================================================
 * Governor callbacks
 * ================================================================ */

static int nap_fallback_heuristic(struct cpuidle_driver *drv,
				  struct cpuidle_device *dev)
{
	s64 latency_req = cpuidle_governor_latency_req(dev->cpu);
	int i;

	for (i = drv->state_count - 1; i > 0; i--) {
		if (dev->states_usage[i].disable)
			continue;
		if (drv->states[i].exit_latency_ns <= latency_req)
			return i;
	}
	return 0;
}

static int nap_select(struct cpuidle_driver *drv,
		      struct cpuidle_device *dev,
		      bool *stop_tick)
{
	struct nap_cpu_data *d = this_cpu_ptr(&nap_data);
	int idx;

	if (unlikely(drv->state_count <= 1))
		return 0;

	if (likely(may_use_simd())) {
		kernel_fpu_begin();
		idx = nap_fpu_select(drv, dev, d);
		kernel_fpu_end();

		if (idx < 0)
			idx = nap_fallback_heuristic(drv, dev);
	} else {
		idx = nap_fallback_heuristic(drv, dev);
	}

	*stop_tick = (drv->states[idx].target_residency_ns >
		      RESIDENCY_THRESHOLD_NS);

	d->last_selected_idx = idx;
	d->stats.total_selects++;

	return idx;
}

static void nap_reflect(struct cpuidle_device *dev, int index)
{
	struct nap_cpu_data *d = this_cpu_ptr(&nap_data);
	struct cpuidle_driver *drv = cpuidle_get_cpu_driver(dev);
	u64 measured_ns = dev->last_residency_ns;

	if (unlikely(!drv))
		return;

	nap_history_update(d, measured_ns);
	nap_update_hit_intercept(d, drv, dev, index, measured_ns);

	d->last_prediction_error = d->last_predicted_ns - (s64)measured_ns;
	nap_update_external_signals(d, dev->cpu);

	if (d->learning_mode && ++d->learn_counter >= d->learn_interval) {
		d->learn_counter = 0;
		d->learn_actual_ns = measured_ns;
		d->needs_learn = true;
	}

	d->stats.total_residency_ns += measured_ns;
	if (index > 0 && measured_ns < drv->states[index].target_residency_ns)
		d->stats.undershoot_count++;
}

static int nap_enable(struct cpuidle_driver *drv,
		      struct cpuidle_device *dev)
{
	struct nap_cpu_data *d = per_cpu_ptr(&nap_data, dev->cpu);

	memset(d, 0, sizeof(*d));

	/*
	 * Defer weight initialization to the first nap_select() FPU path
	 * via reset_pending.  nap_enable() is called from cpuidle core
	 * (cpuidle_enable_device) which may run on a different CPU than
	 * dev->cpu during governor switch.  Deferring ensures FPU init
	 * happens on the correct CPU in its own idle context.
	 */
	WRITE_ONCE(nap_cached_drv, drv);
	d->learning_mode  = true;
	d->learning_rate_millths  = NAP_DEFAULT_LR_MILLTHS;
	d->learn_interval = NAP_DEFAULT_INTERVAL;
	d->max_grad_norm_millths  = NAP_DEFAULT_CLAMP_MILLTHS;
	d->warmup_threshold = NAP_DEFAULT_WARMUP;
	d->convergence_thresh = NAP_DEFAULT_CONVERGE;
	d->reset_pending = true;

	return 0;
}

static void nap_disable(struct cpuidle_driver *drv,
			struct cpuidle_device *dev)
{
	WRITE_ONCE(nap_cached_drv, NULL);
}

/* ================================================================
 * sysfs interface  (/sys/devices/system/cpu/cpuidle/nap/)
 * ================================================================ */

static ssize_t stats_show(struct kobject *kobj,
			  struct kobj_attribute *attr, char *buf)
{
	int cpu, len = 0;
	u64 total_sel = 0, total_res = 0, total_under = 0, total_learn = 0;
	int converged_cpus = 0, online_cpus = 0;

	for_each_online_cpu(cpu) {
		struct nap_cpu_data *d = &per_cpu(nap_data, cpu);

		total_sel   += d->stats.total_selects;
		total_res   += d->stats.total_residency_ns;
		total_under += d->stats.undershoot_count;
		total_learn += d->stats.learn_count;
		if (d->converged)
			converged_cpus++;
		online_cpus++;
	}

	len += sysfs_emit_at(buf, len, "total_selects: %llu\n", total_sel);
	len += sysfs_emit_at(buf, len, "total_residency_ms: %llu\n",
			     div_u64(total_res, NSEC_PER_MSEC));
	len += sysfs_emit_at(buf, len, "undershoot_count: %llu\n", total_under);
	len += sysfs_emit_at(buf, len, "undershoot_rate_permil: %llu\n",
			     total_sel ? div_u64(total_under * 1000, total_sel) : 0);
	len += sysfs_emit_at(buf, len, "learn_count: %llu\n", total_learn);
	len += sysfs_emit_at(buf, len, "converged_cpus: %d/%d\n",
			     converged_cpus, online_cpus);
	return len;
}

static ssize_t learning_mode_show(struct kobject *kobj,
				  struct kobj_attribute *attr, char *buf)
{
	int cpu, learning = 0, converged = 0, total = 0;

	for_each_online_cpu(cpu) {
		struct nap_cpu_data *d = &per_cpu(nap_data, cpu);

		if (d->learning_mode)
			learning++;
		if (d->converged)
			converged++;
		total++;
	}

	if (total == 0)
		return sysfs_emit(buf, "off\n");

	if (learning == 0)
		return sysfs_emit(buf, "off\n");
	else if (converged == total)
		return sysfs_emit(buf, "online\n");
	else
		return sysfs_emit(buf, "warmup (%d/%d converged)\n",
				  converged, total);
}

static ssize_t learning_mode_store(struct kobject *kobj,
				   struct kobj_attribute *attr,
				   const char *buf, size_t count)
{
	bool mode;
	int cpu;

	if (sysfs_streq(buf, "online"))
		mode = true;
	else if (sysfs_streq(buf, "off"))
		mode = false;
	else
		return -EINVAL;

	for_each_online_cpu(cpu)
		per_cpu(nap_data, cpu).learning_mode = mode;

	return count;
}

static ssize_t learning_rate_show(struct kobject *kobj,
				  struct kobj_attribute *attr, char *buf)
{
	int cpu;

	cpu = cpumask_first(cpu_online_mask);
	if (cpu >= nr_cpu_ids)
		return sysfs_emit(buf, "0\n");
	return sysfs_emit(buf, "%u\n",
			  per_cpu(nap_data, cpu).learning_rate_millths);
}

static ssize_t learning_rate_store(struct kobject *kobj,
				   struct kobj_attribute *attr,
				   const char *buf, size_t count)
{
	unsigned int val;
	int cpu;

	if (kstrtouint(buf, 10, &val) || val == 0 || val > 100)
		return -EINVAL;

	for_each_online_cpu(cpu)
		per_cpu(nap_data, cpu).learning_rate_millths = val;

	return count;
}

static ssize_t learn_interval_show(struct kobject *kobj,
				   struct kobj_attribute *attr, char *buf)
{
	int cpu;

	cpu = cpumask_first(cpu_online_mask);
	if (cpu >= nr_cpu_ids)
		return sysfs_emit(buf, "0\n");
	return sysfs_emit(buf, "%d\n",
			  per_cpu(nap_data, cpu).learn_interval);
}

static ssize_t learn_interval_store(struct kobject *kobj,
				    struct kobj_attribute *attr,
				    const char *buf, size_t count)
{
	unsigned int val;
	int cpu;

	if (kstrtouint(buf, 10, &val) || val == 0 || val > 10000)
		return -EINVAL;

	for_each_online_cpu(cpu)
		per_cpu(nap_data, cpu).learn_interval = val;

	return count;
}

static ssize_t reset_weights_store(struct kobject *kobj,
				   struct kobj_attribute *attr,
				   const char *buf, size_t count)
{
	cpumask_var_t mask;
	int cpu;

	if (!READ_ONCE(nap_cached_drv))
		return -ENODEV;

	/*
	 * Set a per-CPU flag; each CPU will reinitialize its own weights
	 * inside nap_select() within its own kernel_fpu_begin/end context.
	 * This avoids cross-CPU data races on the weight arrays.
	 *
	 * Accepts "all" to reset every online CPU, or a cpulist
	 * (e.g. "0-3,5,7") to reset specific CPUs.
	 */
	if (sysfs_streq(buf, "all")) {
		for_each_online_cpu(cpu)
			per_cpu(nap_data, cpu).reset_pending = true;
		pr_info("nap: weight reset scheduled for all CPUs\n");
		return count;
	}

	if (!alloc_cpumask_var(&mask, GFP_KERNEL))
		return -ENOMEM;

	if (cpulist_parse(buf, mask)) {
		free_cpumask_var(mask);
		return -EINVAL;
	}

	for_each_cpu_and(cpu, mask, cpu_online_mask)
		per_cpu(nap_data, cpu).reset_pending = true;

	pr_info("nap: weight reset scheduled for CPUs %*pbl\n",
		cpumask_pr_args(mask));
	free_cpumask_var(mask);
	return count;
}

static ssize_t reset_stats_store(struct kobject *kobj,
				 struct kobj_attribute *attr,
				 const char *buf, size_t count)
{
	int cpu;

	for_each_online_cpu(cpu)
		memset(&per_cpu(nap_data, cpu).stats, 0,
		       sizeof(struct nap_stats));

	return count;
}

static ssize_t warmup_threshold_show(struct kobject *kobj,
				     struct kobj_attribute *attr, char *buf)
{
	int cpu;

	cpu = cpumask_first(cpu_online_mask);
	if (cpu >= nr_cpu_ids)
		return sysfs_emit(buf, "0\n");
	return sysfs_emit(buf, "%u\n",
			  per_cpu(nap_data, cpu).warmup_threshold);
}

static ssize_t warmup_threshold_store(struct kobject *kobj,
				      struct kobj_attribute *attr,
				      const char *buf, size_t count)
{
	unsigned int val;
	int cpu;

	if (kstrtouint(buf, 10, &val) || val > 100000)
		return -EINVAL;

	for_each_online_cpu(cpu)
		per_cpu(nap_data, cpu).warmup_threshold = val;

	return count;
}

static ssize_t convergence_thresh_show(struct kobject *kobj,
				       struct kobj_attribute *attr, char *buf)
{
	int cpu;

	cpu = cpumask_first(cpu_online_mask);
	if (cpu >= nr_cpu_ids)
		return sysfs_emit(buf, "0\n");
	return sysfs_emit(buf, "%u\n",
			  per_cpu(nap_data, cpu).convergence_thresh);
}

static ssize_t convergence_thresh_store(struct kobject *kobj,
					struct kobj_attribute *attr,
					const char *buf, size_t count)
{
	unsigned int val;
	int cpu;

	if (kstrtouint(buf, 10, &val) || val > 1024)
		return -EINVAL;

	for_each_online_cpu(cpu)
		per_cpu(nap_data, cpu).convergence_thresh = val;

	return count;
}

static ssize_t ema_accuracy_show(struct kobject *kobj,
				 struct kobj_attribute *attr, char *buf)
{
	int cpu;
	unsigned int ema_min = 1024, ema_max = 0;
	unsigned long ema_sum = 0;
	int converged = 0, total = 0;

	for_each_online_cpu(cpu) {
		struct nap_cpu_data *d = &per_cpu(nap_data, cpu);

		if (d->ema_accuracy < ema_min)
			ema_min = d->ema_accuracy;
		if (d->ema_accuracy > ema_max)
			ema_max = d->ema_accuracy;
		ema_sum += d->ema_accuracy;
		if (d->converged)
			converged++;
		total++;
	}

	if (total == 0)
		return sysfs_emit(buf, "no cpus online\n");

	return sysfs_emit(buf, "min/avg/max: %u/%lu/%u (x1024)\nconverged: %d/%d\n",
			  ema_min, ema_sum / total, ema_max,
			  converged, total);
}

static ssize_t version_show(struct kobject *kobj,
			    struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%s\n", CPUIDLE_NAP_VERSION);
}

static ssize_t simd_show(struct kobject *kobj,
			 struct kobj_attribute *attr, char *buf)
{
	if (static_branch_unlikely(&nap_use_avx512))
		return sysfs_emit(buf, "avx512\n");
	else if (static_branch_unlikely(&nap_use_avx2))
		return sysfs_emit(buf, "avx2\n");
	else
		return sysfs_emit(buf, "sse2\n");
}

static ssize_t converged_show(struct kobject *kobj,
			      struct kobj_attribute *attr, char *buf)
{
	int cpu, converged = 0, total = 0;

	for_each_online_cpu(cpu) {
		struct nap_cpu_data *d = &per_cpu(nap_data, cpu);

		if (d->converged)
			converged++;
		total++;
	}
	return sysfs_emit(buf, "%d/%d\n", converged, total);
}

static struct kobj_attribute version_attr            = __ATTR_RO(version);
static struct kobj_attribute simd_attr               = __ATTR_RO(simd);
static struct kobj_attribute converged_attr          = __ATTR_RO(converged);
static struct kobj_attribute stats_attr              = __ATTR_RO(stats);
static struct kobj_attribute learning_mode_attr      = __ATTR_RW(learning_mode);
static struct kobj_attribute learning_rate_attr      = __ATTR_RW(learning_rate);
static struct kobj_attribute learn_interval_attr     = __ATTR_RW(learn_interval);
static struct kobj_attribute warmup_threshold_attr   = __ATTR_RW(warmup_threshold);
static struct kobj_attribute convergence_thresh_attr = __ATTR_RW(convergence_thresh);
static struct kobj_attribute ema_accuracy_attr       = __ATTR_RO(ema_accuracy);
static struct kobj_attribute reset_weights_attr      = __ATTR_WO(reset_weights);
static struct kobj_attribute reset_stats_attr        = __ATTR_WO(reset_stats);

static struct attribute *nap_attrs[] = {
	&version_attr.attr,
	&simd_attr.attr,
	&converged_attr.attr,
	&stats_attr.attr,
	&learning_mode_attr.attr,
	&learning_rate_attr.attr,
	&learn_interval_attr.attr,
	&warmup_threshold_attr.attr,
	&convergence_thresh_attr.attr,
	&ema_accuracy_attr.attr,
	&reset_weights_attr.attr,
	&reset_stats_attr.attr,
	NULL,
};

static const struct attribute_group nap_attr_group = {
	.attrs = nap_attrs,
};

static struct kobject *cpuidle_kobj;

int nap_sysfs_init(void)
{
	struct device *dev_root;
	int ret;

	dev_root = bus_get_dev_root(&cpu_subsys);
	if (!dev_root)
		return -ENODEV;

	cpuidle_kobj = kobject_create_and_add("nap", &dev_root->kobj);
	put_device(dev_root);
	if (!cpuidle_kobj)
		return -ENOMEM;

	ret = sysfs_create_group(cpuidle_kobj, &nap_attr_group);
	if (ret) {
		kobject_put(cpuidle_kobj);
		cpuidle_kobj = NULL;
	}
	return ret;
}

void nap_sysfs_exit(void)
{
	if (cpuidle_kobj) {
		sysfs_remove_group(cpuidle_kobj, &nap_attr_group);
		kobject_put(cpuidle_kobj);
		cpuidle_kobj = NULL;
	}
}

/* ================================================================
 * Governor registration
 * ================================================================ */

static struct cpuidle_governor nap_governor = {
	.name    = "nap",
	.rating  = 26,
	.enable  = nap_enable,
	.disable = nap_disable,
	.select  = nap_select,
	.reflect = nap_reflect,
};

static int __init nap_init(void)
{
	int ret;

	nap_detect_simd();

	ret = nap_sysfs_init();
	if (ret)
		pr_warn("nap: sysfs init failed: %d (continuing without sysfs)\n", ret);

	ret = cpuidle_register_governor(&nap_governor);
	if (ret) {
		pr_err("nap: register_governor failed: %d\n", ret);
		nap_sysfs_exit();
		return ret;
	}

	pr_info("%s v%s by %s registered (rating=%u)\n",
	       CPUIDLE_NAP_PROGNAME, CPUIDLE_NAP_VERSION,
	       CPUIDLE_NAP_AUTHOR, nap_governor.rating);
	return 0;
}
postcore_initcall(nap_init);
