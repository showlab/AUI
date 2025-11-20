"""
Transformation and analysis helpers for CUA Failure Revision.
Includes storyboard-based commenter calls, signatures, and analysis merging.
"""

import json
import time
import asyncio
import hashlib
from typing import Dict, Any, List
from pathlib import Path


def compute_initial_signature(comp, app_name: str, model_name: str, v0_html: str, v0_dir: str = None) -> str:
    h = hashlib.sha256()
    scope = v0_dir if v0_dir else "default"
    h.update(scope.encode('utf-8'))
    h.update(app_name.encode('utf-8'))
    h.update(model_name.encode('utf-8'))
    h.update(v0_html.encode('utf-8'))
    return h.hexdigest()


def compute_legacy_initial_signature(comp, app_name: str, model_name: str, v0_html: str, v0_dir: str = None) -> str:
    h = hashlib.sha256()
    h.update(app_name.encode('utf-8'))
    h.update(model_name.encode('utf-8'))
    h.update(v0_html.encode('utf-8'))
    return h.hexdigest()


async def generate_single_comment(comp, app_name: str, model_name: str, task_index: int,
                                  task_description: str, expected_outcome: str,
                                  trajectory_dir: str = None, v0_dir: str = None,
                                  v0_html: str = None, cache_file: Path = None,
                                  commenter_model: str = None, progress_tracker=None) -> str:
    try:
        from utils.storyboard_generator import generate_failure_storyboard
        storyboard_start = time.time()
        # trajectory_dir may be provided by caller; if missing, derive from initial layout
        if trajectory_dir is None:
            # Best-effort derivation matching Stage 2 directory structure
            base_dir = comp._base_dir
            scope = v0_dir if v0_dir else "default"
            # Default to uitars as source CUA model when not specified explicitly
            trajectory_dir = (base_dir / "initial" / scope / "tasks" / app_name /
                              "initial_cua_results" / model_name / "uitars" /
                              "trajectories" / f"task_{task_index}")
        else:
            trajectory_dir = Path(trajectory_dir)
            storyboard_result = await generate_failure_storyboard(
                app_name=app_name,
                model_name=model_name,
                task_index=task_index,
                task_description=task_description,
                expected_outcome=expected_outcome,
                trajectory_dir=trajectory_dir,
                v0_dir=v0_dir
            )
        storyboard_time = time.time() - storyboard_start
        if not storyboard_result:
            return f"FAILED: storyboard_failed task {task_index}"
        storyboard_path = Path(storyboard_result)

        comment_start = time.time()
        analysis_model = commenter_model or model_name

        analysis = await comp.commenter.analyze_single_failure(
            storyboard_path=str(storyboard_path),
            html_content=v0_html,
            model_name=analysis_model
        )
        comment_time = time.time() - comment_start
        if progress_tracker:
            timing_info = f"Task {task_index}: Storyboard {storyboard_time:.1f}s, Comment {comment_time:.1f}s"
            progress_tracker.add_timing_info(model_name, "BATCH", timing_info)

        if not analysis or len(analysis.strip()) < 10:
            return f"FAILED: empty_or_invalid_analysis task {task_index}"

        if cache_file is not None:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(analysis.strip(), encoding='utf-8')
            except Exception:
                pass
        return f"Task {task_index} Analysis:\n{analysis}"

    except Exception as e:
        error_type = "API_ERROR" if ("429" in str(e) or "401" in str(e)) else "COMMENTER_ERROR"
        error_detail = str(e).replace('\n', ' ')[:100]
        if progress_tracker:
            progress_tracker.add_timing_info(model_name, "BATCH", f"Task {task_index} FAILED: {error_type} - {error_detail}")
        return f"FAILED: commenter_error task {task_index} ({error_type}) {error_detail}"


async def load_task_descriptions(comp, app_name: str, v0_dir: str = None) -> Dict[int, Dict[str, Any]]:
    if v0_dir:
        tasks_file = comp._base_dir / "initial" / v0_dir / "tasks" / app_name / "tasks.json"
    else:
        tasks_file = Path(f"tasks/{app_name}/tasks.json")
    task_descriptions = {}
    if tasks_file.exists():
        with open(tasks_file, 'r', encoding='utf-8') as f:
            tasks_data = json.load(f)
            for task in tasks_data.get('tasks', []):
                task_descriptions[task['id']] = {
                    'description': task.get('description', ''),
                    'expected_outcome': task.get('expected_outcome', ''),
                    'category': task.get('category', ''),
                    'complexity': task.get('complexity', '')
                }
    return task_descriptions


def combine_failure_analyses(failure_analyses: List[str]) -> str:
    if not failure_analyses:
        return "No failure analyses available"
    if len(failure_analyses) == 1:
        return failure_analyses[0]
    combined = "## Combined Failure Analysis\n\n"
    combined += f"Analyzed {len(failure_analyses)} failed trajectories:\n\n"
    for i, analysis in enumerate(failure_analyses, 1):
        combined += f"### Failure {i}\n{analysis}\n\n"
    combined += "## Summary\n"
    combined += "Multiple UI design issues were identified across the failed trajectories. "
    combined += "The following improvements should address the common failure patterns identified above."
    return combined


async def generate_all_comments(comp, app_name: str, model_name: str,
                                failed_tasks: List[Dict[str, Any]], v0_html: str,
                                v0_dir: str = None, progress_tracker=None,
                                force: bool = False) -> List[str]:
    stage1_start = time.time()
    if progress_tracker:
        progress_tracker.add_timing_info(model_name, app_name, f"Stage 1: Started {len(failed_tasks)} comment tasks")
    task_descriptions = await load_task_descriptions(comp, app_name, v0_dir)
    cache_scope = v0_dir if v0_dir else "default"
    cache_base = comp._base_dir / "initial" / cache_scope / "comments" / comp._commenter_variant

    comment_tasks = []
    valid_analyses = []
    for i, task in enumerate(failed_tasks):
        task_index = task.get('task_index', 0)
        trajectory_dir = task.get('trajectory_dir')
        cache_file = cache_base / model_name / app_name / f"task_{task_index}.txt"
        task_description = task_descriptions.get(task_index, {}).get('description', 'Unknown task')
        expected_outcome = task_descriptions.get(task_index, {}).get('expected_outcome', 'Unknown outcome')
        if cache_file.exists() and not force:
            try:
                cached_text = cache_file.read_text(encoding='utf-8')
                valid_analyses.append(f"Task {task_index} Analysis:\n{cached_text}")
                continue
            except Exception:
                pass

        async def limited_comment(task_index=task_index, trajectory_dir=trajectory_dir,
                                  task_description=task_description, expected_outcome=expected_outcome,
                                  cache_file=cache_file, v0_html=v0_html, model_name=model_name,
                                  app_name=app_name, progress_tracker=progress_tracker):
            async with comp._commenter_sema:
                from .cua_failure_transform import generate_single_comment
                return await generate_single_comment(
                    comp=comp,
                    app_name=app_name,
                    model_name=model_name,
                    task_index=task_index,
                    task_description=task_description,
                    expected_outcome=expected_outcome,
                    trajectory_dir=trajectory_dir,
                    v0_dir=v0_dir,
                    v0_html=v0_html,
                    cache_file=cache_file,
                    commenter_model=model_name,
                    progress_tracker=progress_tracker,
                )

        comment_tasks.append(limited_comment())

    if progress_tracker:
        progress_tracker.update_status(model_name, app_name, f"📝 Stage 1: 0/{len(comment_tasks)} done")

    if comment_tasks:
        completed = 0
        for fut in asyncio.as_completed(comment_tasks):
            try:
                analysis = await fut
                completed += 1
                if analysis and (analysis.startswith("FAILED:")):
                    if progress_tracker:
                        progress_tracker.add_timing_info(model_name, app_name, analysis)
                elif analysis and len(analysis.strip()) > 10:
                    valid_analyses.append(analysis)
                if progress_tracker:
                    progress_tracker.update_status(model_name, app_name, f"📝 Stage 1: {completed}/{len(comment_tasks)} done")
            except Exception:
                completed += 1
                if progress_tracker:
                    progress_tracker.update_status(model_name, app_name, f"📝 Stage 1: {completed}/{len(comment_tasks)} done")

    stage1_time = time.time() - stage1_start
    if progress_tracker:
        progress_tracker.add_timing_info(model_name, app_name, f"Stage 1: Completed in {stage1_time:.1f}s")
    return valid_analyses


async def generate_all_comments_batch(comp, model_name: str, all_failed_tasks: List[Dict[str, Any]],
                                      v0_dir: str = None, progress_tracker=None,
                                      force: bool = False, suppress: bool = True) -> List[str]:
    unique_apps = sorted(set(task['source_app'] for task in all_failed_tasks))
    app_descriptions = {}
    for app_name in unique_apps:
        app_descriptions[app_name] = await load_task_descriptions(comp, app_name, v0_dir)

    cache_scope = v0_dir if v0_dir else "default"
    cache_base = comp._base_dir / "initial" / cache_scope / "comments" / comp._commenter_variant

    if progress_tracker:
        msg = f"🚀 Creating {len(all_failed_tasks)} comment tasks..."
        if suppress:
            progress_tracker.add_timing_info(model_name, "BATCH", msg)
        else:
            progress_tracker.update_status(model_name, "BATCH", msg)

    total_tasks = len(all_failed_tasks)
    cached_count = 0
    cached_ok_keys = set()
    cached_analyses = []

    try:
        from pathlib import Path as _P
        results_root = (comp._base_dir / "initial" / v0_dir / "tasks") if v0_dir else _P("tasks")
        for app_name in unique_apps:
            res_path = results_root / app_name / 'initial_cua_results' / model_name / 'results.json'
            if not res_path.exists():
                continue
            try:
                data = json.loads(res_path.read_text(encoding='utf-8'))
                failed = [int(tr.get('task_index', 0)) for tr in data.get('task_results', []) if not tr.get('completed', False)]
            except Exception:
                failed = []
            for task_index in failed:
                cache_file = cache_base / model_name / app_name / f"task_{task_index}.txt"
                if cache_file.exists() and not force:
                    try:
                        cached_text = cache_file.read_text(encoding='utf-8')
                        cached_analyses.append(f"Task {task_index} Analysis:\n{cached_text}")
                        cached_ok_keys.add((app_name, task_index))
                        cached_count += 1
                    except Exception:
                        pass
    except Exception:
        for task in all_failed_tasks:
            app_name = task['source_app']
            task_index = task.get('task_index', 0)
            cache_file = cache_base / model_name / app_name / f"task_{task_index}.txt"
            if cache_file.exists() and not force:
                try:
                    cached_text = cache_file.read_text(encoding='utf-8')
                    cached_analyses.append(f"Task {task_index} Analysis:\n{cached_text}")
                    cached_ok_keys.add((app_name, task_index))
                    cached_count += 1
                except Exception:
                    pass

    if progress_tracker:
        scheduled = total_tasks - cached_count
        msg = f"📝 Stage 1: cached {cached_count}/{total_tasks} | done 0/{scheduled}"
        if suppress:
            progress_tracker.add_timing_info(model_name, "BATCH", msg)
        else:
            progress_tracker.update_status(model_name, "BATCH", msg)
        progress_tracker.add_timing_info(
            model_name, "BATCH",
            f"Stage 1: Using cache for {cached_count}/{total_tasks} tasks; scheduling {scheduled}"
        )

    comment_tasks = []
    for i, task in enumerate(all_failed_tasks):
        app_name = task['source_app']
        task_index = task.get('task_index', 0)
        v0_html = task['v0_html']
        task_desc = app_descriptions[app_name].get(task_index, {})
        if (not force) and ((app_name, task_index) in cached_ok_keys):
            continue

        async def limited_comment(app_name=app_name, model_name=model_name, task_index=task_index,
                                  task=task, task_desc=task_desc, v0_html=v0_html, v0_dir=v0_dir,
                                  progress_tracker=progress_tracker, force=force):
            async with comp._commenter_sema:
                return await generate_single_comment(
                    comp=comp,
                    app_name=app_name,
                    model_name=model_name,
                    task_index=task_index,
                    task_description=task.get('description', 'Unknown task'),
                    expected_outcome=task_desc.get('expected_outcome', 'Unknown outcome'),
                    trajectory_dir=task.get('trajectory_dir'),
                    v0_html=v0_html,
                    v0_dir=v0_dir,
                    commenter_model=model_name,
                    progress_tracker=progress_tracker,
                )

        comment_tasks.append(limited_comment())

    cached_count_final = total_tasks - len(comment_tasks)
    scheduled = len(comment_tasks)
    if progress_tracker:
        msg = f"📝 Stage 1: cached {cached_count_final}/{total_tasks} | done 0/{scheduled}"
        if suppress:
            progress_tracker.add_timing_info(model_name, "BATCH", msg)
        else:
            progress_tracker.update_status(model_name, "BATCH", msg)

    valid_analyses = list(cached_analyses)
    if comment_tasks:
        if progress_tracker:
            msg = f"🔄 Stage 1: Starting {scheduled} comment tasks..."
            if suppress:
                progress_tracker.add_timing_info(model_name, "BATCH", msg)
            else:
                progress_tracker.update_status(model_name, "BATCH", msg)
        try:
            completed_count = 0
            failed_count = 0
            for completed_task in asyncio.as_completed(comment_tasks):
                try:
                    analysis = await completed_task
                    completed_count += 1
                    if analysis and (analysis.startswith("FAILED:")):
                        failed_count += 1
                        if progress_tracker:
                            progress_tracker.add_timing_info(model_name, "BATCH", analysis)
                    elif analysis and len(analysis.strip()) > 10:
                        valid_analyses.append(analysis)
                        if progress_tracker and "Analysis:" in analysis:
                            if "\n" in analysis:
                                analysis_content = analysis.split('\n', 1)[1]
                                progress_tracker.add_analysis_info(model_name, "BATCH", analysis_content)
                    if progress_tracker and not suppress:
                        suffix = f" | failed {failed_count}" if failed_count > 0 else ""
                        progress_tracker.update_status(
                            model_name, "BATCH",
                            f"📝 Stage 1: cached {cached_count_final}/{total_tasks} | done {completed_count}/{scheduled}{suffix}"
                        )
                except Exception:
                    failed_count += 1
                    completed_count += 1
                    if progress_tracker and not suppress:
                        progress_tracker.update_status(model_name, "BATCH",
                                                       f"📝 Stage 1: {completed_count}/{total_tasks} ({len(valid_analyses)} valid, {failed_count} failed)")
            if progress_tracker:
                msg = f"✅ Stage 1: scheduled {scheduled} done; cached {cached_count_final}; failed {failed_count}"
                if suppress:
                    progress_tracker.add_timing_info(model_name, "BATCH", msg)
                else:
                    progress_tracker.update_status(model_name, "BATCH", msg)
        except Exception as e:
            if progress_tracker:
                error_detail = str(e).replace('\n', ' ')[:100]
                progress_tracker.add_timing_info(model_name, "BATCH", f"Stage 1 execution FAILED: {error_detail}")
    else:
        if progress_tracker:
            msg = f"📝 Stage 1: cached {total_tasks}/{total_tasks} | done 0/0"
            if suppress:
                progress_tracker.add_timing_info(model_name, "BATCH", msg)
            else:
                progress_tracker.update_status(model_name, "BATCH", msg)
    return valid_analyses
