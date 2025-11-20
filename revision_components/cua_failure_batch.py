"""
Batch and orchestration helpers for CUA Failure Revision.
These functions coordinate multi-app processing and per-app revised generation.
"""

import json
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, List

from .cua_failure_transform import (
    combine_failure_analyses,
    generate_all_comments,
    generate_all_comments_batch,
    compute_initial_signature,
    compute_legacy_initial_signature,
)


async def process_model_batch(comp, model_name: str, app_batch_data: List[Dict[str, Any]],
                              destylized: bool, v0_dir: str = None, progress_tracker=None,
                              force_comments: bool = False, force_v1: bool = False,
                              non_regression_prompts_by_app: Dict[str, str] = None) -> Dict[str, Any]:
    batch_start = time.time()

    all_failed_tasks = []
    for app_data in app_batch_data:
        app_name = app_data['app_name']
        failed_tasks = app_data.get('failed_tasks', [])
        for task in failed_tasks:
            task_with_app = {**task, 'source_app': app_name, 'v0_html': app_data['v0_html']}
            all_failed_tasks.append(task_with_app)

    if progress_tracker:
        progress_tracker.update_status(model_name, "BATCH",
                                       f"📝 Stage 1: Processing {len(all_failed_tasks)} failed tasks from {len(app_batch_data)} apps...")
        progress_tracker.add_timing_info(model_name, "BATCH",
                                         f"Starting Stage 1 with {len(all_failed_tasks)} tasks")

    stage1_start = time.time()
    all_analyses = []
    if comp.commenter and all_failed_tasks:
        all_analyses = await generate_all_comments_batch(
            comp, model_name, all_failed_tasks, v0_dir, progress_tracker, force_comments
        )
    stage1_time = time.time() - stage1_start
    if progress_tracker:
        valid = len(all_analyses)
        total = len(all_failed_tasks)
        failed = total - valid
        progress_tracker.update_status(model_name, "BATCH",
                                       f"✅ Stage 1: valid {valid}/{total} | failed {failed} in {stage1_time:.1f}s")
        progress_tracker.add_timing_info(model_name, "BATCH",
                                         f"Stage 1: valid {valid}/{total}, failed {failed} in {stage1_time:.1f}s")

    stage2_start = time.time()
    if progress_tracker:
        progress_tracker.update_status(model_name, "BATCH", f"✏️ Stage 2: 0/{len(app_batch_data)} websites")


    async def run_single_app(app_data):
        app_name_l = app_data['app_name']
        v0_html_l = app_data['v0_html']
        failed_tasks_l = app_data.get('failed_tasks', [])

        if progress_tracker:
            progress_tracker.add_timing_info(model_name, "BATCH", f"{app_name_l}: ✏️ Stage 2: Generating revised website...")

        if failed_tasks_l:
            app_task_indexes = {t.get('task_index', -1) for t in failed_tasks_l}
            app_analyses = [analysis for analysis in all_analyses
                            if any(analysis.startswith(f"Task {idx}") for idx in app_task_indexes)]
            combined_analysis = combine_failure_analyses(app_analyses)
        else:
            app_analyses = []
            combined_analysis = None

        v1_cache_dir = comp._revised_cache_dir(comp._base_dir, v0_dir, comp._revision_variant, model_name, app_name_l)
        v1_html_path = v1_cache_dir / "index.html"
        v1_meta_path = v1_cache_dir / "meta.json"
        v0_sig = compute_initial_signature(comp, app_name_l, model_name, v0_html_l, v0_dir)
        legacy_sig = compute_legacy_initial_signature(comp, app_name_l, model_name, v0_html_l, v0_dir)

        if (not force_v1) and v1_html_path.exists() and v1_meta_path.exists():
            try:
                meta = json.loads(v1_meta_path.read_text(encoding='utf-8'))
                sig_in_meta = meta.get('v0_signature')
                if sig_in_meta == v0_sig or sig_in_meta == legacy_sig:
                    cached_html = v1_html_path.read_text(encoding='utf-8')
                    if progress_tracker:
                        progress_tracker.add_timing_info(model_name, "BATCH", f"{app_name_l}: ✅ Using cached revised website")
                    return {'app_name': app_name_l, 'result': {
                        'success': True,
                        'html_content': cached_html,
                        'analyzed_failures': len(app_analyses),
                        'revision_type': 'cua_failure',
                        'destylized': destylized,
                        'cache_used': True,
                        'v0_signature': v0_sig
                    }}
            except Exception:
                pass


        try:
            if failed_tasks_l:
                result = await comp.coder.generate_revised_website(
                    model_name=model_name,
                    app_name=app_name_l,
                    v0_html=v0_html_l,
                    failed_tasks=failed_tasks_l,
                    failure_analysis=combined_analysis,
                    apply_destylization=destylized,
                    v0_dir=v0_dir,
                    progress_tracker=progress_tracker,
                    verbosity=("high" if model_name == 'gpt5' else None),
                    reasoning_effort=("high" if model_name == 'gpt5' else None),
                    non_regression_contract_prompt=(non_regression_prompts_by_app or {}).get(app_name_l, "")
                )
                if progress_tracker and 'retry_details' in result and result['retry_details']:
                    details = result['retry_details']
                    if isinstance(details, list):
                        for at in details:
                            attempt = at.get('attempt', '?')
                            success = at.get('success', False)
                            gen_time = at.get('generation_time', None)
                            html_len = at.get('html_length', None)
                            icon = '✅' if success else '❌'
                            parts = [f"{app_name_l} Attempt {attempt}"]
                            if gen_time is not None:
                                parts.append(f"{gen_time}s")
                            if html_len is not None:
                                parts.append(f"{html_len} chars")
                            progress_tracker.add_timing_info(model_name, "BATCH", f"RETRY {icon}: {' '.join(parts)}")
                    else:
                        progress_tracker.add_timing_info(model_name, "BATCH", f"RETRY: {app_name_l} {str(details)}")
                if result.get('success') and result.get('html_content'):
                    try:
                        v1_cache_dir.mkdir(parents=True, exist_ok=True)
                        v1_html_path.write_text(result['html_content'], encoding='utf-8')
                        meta = {
                            'app': app_name_l,
                            'model': model_name,
                            'v0_dir': (v0_dir if v0_dir else 'default'),
                            'v0_signature': v0_sig,
                            'generated_at': datetime.now().isoformat(),
                            'analyzed_failures': len(app_analyses)
                        }
                        v1_meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
                    except Exception:
                        pass
                result['analyzed_failures'] = len(app_analyses)
                result['revision_type'] = 'cua_failure'
                result['destylized'] = destylized
                result['v0_signature'] = v0_sig
            else:
                result = await comp.coder.generate_revised_website(
                    model_name=model_name,
                    app_name=app_name_l,
                    v0_html=v0_html_l,
                    failed_tasks=[],
                    failure_analysis="No CUA failures available; apply destylization and fit within screen.",
                    apply_destylization=destylized,
                    v0_dir=v0_dir,
                    progress_tracker=progress_tracker,
                    non_regression_contract_prompt=(non_regression_prompts_by_app or {}).get(app_name_l, "")
                )
                result['analyzed_failures'] = 0
                result['revision_type'] = 'cua_failure'
                result['destylized'] = destylized
                result['v0_signature'] = v0_sig
        finally:
            pass

        return {'app_name': app_name_l, 'result': result}

    total_apps = len(app_batch_data)
    app_results = []
    done = 0
    stage2_tasks = [run_single_app(app_data) for app_data in app_batch_data]
    for fut in asyncio.as_completed(stage2_tasks):
        res = await fut
        app_results.append(res)
        done += 1
        if progress_tracker:
            progress_tracker.update_status(model_name, "BATCH", f"✏️ Stage 2: {done}/{total_apps} websites")

    stage2_time = time.time() - stage2_start
    batch_time = time.time() - batch_start

    if progress_tracker:
        progress_tracker.update_status(model_name, "BATCH", f"🎉 Completed: {len(app_results)} websites in {batch_time:.1f}s total")
        progress_tracker.add_timing_info(model_name, "BATCH",
                                         f"Stage 2: Generated {len(app_results)} websites in {stage2_time:.1f}s")
        progress_tracker.add_timing_info(model_name, "BATCH",
                                         f"Total batch time: {batch_time:.1f}s")

    successful_apps = sum(1 for app_result in app_results if app_result['result'].get('success', False))
    return {
        'success': successful_apps > 0,
        'model': model_name,
        'processed_apps': len(app_batch_data),
        'successful_apps': successful_apps,
        'total_failed_tasks': len(all_failed_tasks),
        'total_analyses': len(all_analyses),
        'batch_time': batch_time,
        'app_results': app_results
    }


async def generate_revised_version(comp, model_name: str, app_name: str, v0_html: str,
                                   failed_tasks: List[Dict[str, Any]], destylized: bool,
                                   v0_dir: str = None, progress_tracker=None,
                                   contract_prompt: str = None) -> Dict[str, Any]:
    start_time = time.time()
    failure_analyses = []
    if comp.commenter:
        if progress_tracker:
            progress_tracker.update_status(model_name, app_name, f"📝 Stage 1: Generating comments...")
        failure_analyses = await generate_all_comments(
            comp, app_name, model_name, failed_tasks, v0_html, v0_dir, progress_tracker
        )

    if progress_tracker:
        progress_tracker.add_timing_info(model_name, app_name, f"Starting to combine {len(failure_analyses)} analyses")
    combined_analysis = combine_failure_analyses(failure_analyses)

    if progress_tracker:
        progress_tracker.update_status(model_name, app_name, f"✏️ Stage 2: Generating revised website...")
    progress_tracker.add_timing_info(model_name, app_name, f"Starting coder.generate_revised_website call")
    result = await comp.coder.generate_revised_website(
        model_name=model_name,
        app_name=app_name,
        v0_html=v0_html,
        failed_tasks=failed_tasks,
        failure_analysis=combined_analysis,
        apply_destylization=destylized,
        v0_dir=v0_dir,
        progress_tracker=progress_tracker,
        non_regression_contract_prompt=contract_prompt
    )
    generation_time = time.time() - start_time
    result['revision_type'] = 'cua_failure'
    result['destylized'] = destylized
    result['analyzed_failures'] = len(failure_analyses)
    result['generation_time'] = generation_time
    result['failure_analysis'] = combined_analysis
    return result
