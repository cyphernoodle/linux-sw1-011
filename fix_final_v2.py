import sys
import re

def fix_file(filename, func):
    with open(filename, 'r') as f:
        content = f.read()
    new_content = func(content)
    with open(filename, 'w') as f:
        f.write(new_content)

def fix_debug(content):
    # Remove the conflicting redeclarations
    content = content.replace('/* static unsigned int sysctl_sched_min_base_slice = 0; */\n/* static void sched_update_min_base_slice(void) { } */', '')
    return content

def fix_fair(content):
    # Fix yield_task_fair corruption
    content = re.sub(
        r'update_curr\(cfs_rq\);\n#ifdef CONFIG_SCHED_BORE\n\trestart_burst_rescale_deadline_bore\(rq->curr\);\n\tclear_buddies\(cfs_rq, &rq->curr->se\);\n#endif /\* CONFIG_SCHED_BORE \*/\n#ifdef CONFIG_SCHED_BORE\n\trestart_burst_rescale_deadline_bore\(curr\);\n\t#if !defined\(CONFIG_SCHED_BORE\)\n\tif \(unlikely\(rq->nr_running == 1\)\)\n\t\treturn;\n\n\tclear_buddies\(cfs_rq, se\);\n#endif /\* CONFIG_SCHED_BORE \*/\n\n\tclear_buddies\(cfs_rq, se\);\n#endif /\* CONFIG_SCHED_BORE \*/',
        r'update_curr(cfs_rq);\n#ifdef CONFIG_SCHED_BORE\n\trestart_burst_rescale_deadline_bore(rq->curr);\n\tclear_buddies(cfs_rq, &rq->curr->se);\n#endif /* CONFIG_SCHED_BORE */',
        content, flags=re.DOTALL
    )

    # Ensure NULL checks for cfs_rq->curr in hooks
    content = content.replace(
        '#ifdef CONFIG_SCHED_BORE\n\tif (cfs_rq->curr && entity_is_task(cfs_rq->curr))\n\t\trestart_burst_rescale_deadline_bore(task_of(cfs_rq->curr));\n\tclear_buddies(cfs_rq, cfs_rq->curr);\n#endif',
        '#ifdef CONFIG_SCHED_BORE\n\tif (cfs_rq->curr && entity_is_task(cfs_rq->curr)) {\n\t\trestart_burst_rescale_deadline_bore(task_of(cfs_rq->curr));\n\t\tclear_buddies(cfs_rq, cfs_rq->curr);\n\t}\n#endif'
    )

    return content

fix_file('kernel/sched/debug.c', fix_debug)
fix_file('kernel/sched/fair.c', fix_fair)
