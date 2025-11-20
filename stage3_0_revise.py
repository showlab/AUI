#!/usr/bin/env python3
"""
Stage 3: 基于失败生成修订版网站
根据初始网站的失败任务与不支持任务生成改进后的修订版
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.append(str(THIS_DIR))

from utils.logging_utils import ts_print
from utils.non_regression import extract_contract, format_contract_prompt, validate_revised, save_json
from utils.model_client import ModelClient
from utils.parallel_runner import ParallelRunner
from agents.coder import Coder
from agents.commenter import Commenter
from agents.commenter_text_only import CommenterTextOnly
from agents.commenter_screenshot_only import CommenterScreenshotOnly
from revision_components.factory import RevisionComponentFactory
from utils.cache_paths import commenter_variant_from_instance
from utils.run_key import build_run_key, short_run_key
from revision_components.revise_runner import build_variant_name as _generate_variant_name
from utils.constants import DEFAULT_APPS

async def revise_model_batch_task(model_name: str, app_name: str, progress_tracker, 
                                 experiment_name: str = "exp1", revision_type: str = "cua", 
                                 destylized: bool = False, v0_dir: str = None, 
                                 commenter_concurrent: int = 10, model_app_groups: dict = None,
                                 force_comments: bool = False,
                                 force_v1: bool = False,
                                 source_cua_model: str = "uitars",
                                 commenter: str = "none",
                                 run_key: str = None,
                                 **kwargs) -> dict:
    """Process all apps for a single model in batch with direct concurrency control"""
    variant_name = _generate_variant_name(revision_type, destylized)
    
    # Get app list for this model from model_app_groups
    app_list = model_app_groups.get(model_name, [])
    if not app_list:
        return {'success': False, 'error': 'No apps for model', 'model': model_name}
    
    try:
        model_client = ModelClient()
        coder = Coder(model_client)
        
        # Initialize commenter via explicit commenter variant
        commenter_inst = None
        if commenter == 'full':
            commenter_inst = Commenter(model_client)
        elif commenter == 'cua-text-only':
            commenter_inst = CommenterTextOnly(model_client)
        elif commenter == 'cua-screenshot-only':
            commenter_inst = CommenterScreenshotOnly(model_client)
        
        # Map CLI revision type to factory type
        factory_revision_type = {
            'cua': 'cua_failure',
            'cua-text-only': 'cua_failure',
            'cua-screenshot-only': 'cua_failure',
            'unsupported': 'unsupported', 
            'integrated': 'integrated'
        }.get(revision_type, revision_type)
        
        # Create revision component with max_concurrent for ALL tasks across apps
        revision_component = RevisionComponentFactory.create_component(
            factory_revision_type,
            coder,
            commenter_inst,
            max_concurrent=commenter_concurrent,
            revision_variant=variant_name,
            commenter_variant=commenter_variant_from_instance(commenter_inst)
        )
        
        # Initialize per-app grid cells for unified display
        for a in app_list:
            progress_tracker.update_status(model_name, a, "📋 Loading...")

        # Use the virtual app key "BATCH" for batch-only timing logs (may be hidden in grid)
        progress_tracker.update_status(model_name, "BATCH", f"Loading batch data... ({len(app_list)} apps)")

        # Adoption of legacy caches is intentionally not handled inline.
        # Use a one-off script to add manifests/meta for existing runs.

        # Prepare batch data for all apps
        app_batch_data = []
        contracts_by_app = {}
        missing_or_invalid = []
        for app_name in app_list:
            app_data = await _load_app_data(model_name, app_name, revision_type, v0_dir, progress_tracker, source_cua_model)
            if app_data:
                # Build non-regression contract per app
                if v0_dir:
                    rules_path = Path(f"initial/{v0_dir}/tasks/{app_name}/states/{model_name}/rules.json")
                else:
                    rules_path = Path(f"tasks/{app_name}/states/{model_name}/rules.json")
                try:
                    contract = extract_contract(rules_path, app_data['v0_html'])
                    contract_prompt = format_contract_prompt(contract)
                except Exception:
                    contract, contract_prompt = ({}, "")
                app_data['non_regression_contract'] = contract
                app_data['non_regression_contract_prompt'] = contract_prompt
                contracts_by_app[app_name] = {
                    'contract': contract,
                    'prompt': contract_prompt,
                }
                app_batch_data.append(app_data)
                progress_tracker.update_status(model_name, app_name, "📦 Context ready")
            else:
                missing_or_invalid.append(app_name)
                progress_tracker.update_status(model_name, app_name, "❌ Missing data", error_detail=f"Missing initial website or context for {app_name}")
        
        if not app_batch_data:
            # Persist visible error line in the grid + timing log
            progress_tracker.update_status(model_name, "BATCH", "❌ No valid app data loaded", error_detail='No valid app data loaded')
            return {
                'success': False,
                'model': model_name,
                'error': 'No valid app data loaded',
                'processed_apps': 0,
                'app_results': []
            }
        
        # Update grid for apps that will be revised
        for a in [x['app_name'] for x in app_batch_data]:
            progress_tracker.update_status(model_name, a, f"✏️ Revising...")
        progress_tracker.update_status(model_name, "BATCH", f"Starting batch revision... ({len(app_batch_data)} apps)")
        
        # Batch revision (no fallback to per-app)
        batch_result = await revision_component.revise_model_batch(
            model_name=model_name,
            app_batch_data=app_batch_data,
            destylized=destylized,
            v0_dir=v0_dir,
            progress_tracker=progress_tracker,
            force_comments=force_comments,
            force_v1=force_v1,
            non_regression_prompts_by_app={a: contracts_by_app.get(a, {}).get('prompt', '') for a in [x['app_name'] for x in app_batch_data]}
        )

        # Save revised websites for each app and update per-app grid status
        for app_result in batch_result.get('app_results', []):
            app_name = app_result['app_name']
            result = app_result['result']
            # Update per-app status in the grid
            if result.get('success'):
                progress_tracker.update_status(model_name, app_name, "💾 Saving revised...")
                if 'retry_details' in result and result['retry_details']:
                    progress_tracker.update_status(model_name, app_name, "✅ Generated revised", retry_info=result['retry_details'])
                else:
                    progress_tracker.update_status(model_name, app_name, "✅ Generated revised")
            else:
                err = result.get('error', 'Unknown error')
                progress_tracker.update_status(model_name, app_name, f"❌ Failed: {err}", error_detail=err)
            # Persist retry details and errors to timing log under BATCH
            if 'retry_details' in result and result['retry_details']:
                details = result['retry_details']
                if isinstance(details, list):
                    for at in details:
                        attempt = at.get('attempt', '?')
                        success = at.get('success', False)
                        gen_time = at.get('generation_time', None)
                        html_len = at.get('html_length', None)
                        icon = '✅' if success else '❌'
                        parts = [f"{app_name} Attempt {attempt}"]
                        if gen_time is not None:
                            parts.append(f"{gen_time}s")
                        if html_len is not None:
                            parts.append(f"{html_len} chars")
                        progress_tracker.add_timing_info(model_name, "BATCH", f"RETRY {icon}: {' '.join(parts)}")
                else:
                    progress_tracker.add_timing_info(model_name, "BATCH", f"RETRY: {app_name} {str(details)}")

            if not result.get('success', False):
                err = result.get('error', 'Unknown error')
                progress_tracker.add_timing_info(model_name, "BATCH", f"ERROR: {app_name} {err}")
            if result.get('success') and result.get('html_content'):
                meta = {
                    'model': model_name,
                    'app': app_name,
                    'variant': variant_name,
                    'v0_signature': result.get('v0_signature'),
                    'analyzed_failures': result.get('analyzed_failures'),
                    'destylized': result.get('destylized'),
                    'generated_at': __import__('datetime').datetime.now().isoformat()
                }
                v1_path = _save_revised_website(
                    result['html_content'], app_name, model_name,
                    experiment_name, run_key, meta
                )
                # Non-regression: save contract + validate revised
                try:
                    contract_pkg = contracts_by_app.get(app_name, {})
                    contract = contract_pkg.get('contract', {})
                    contract_dir = Path(f"experiments/{experiment_name}/runs/{run_key}/stage3_0/{app_name}/{model_name}/non_regression")
                    save_json(contract_dir / "contract.json", contract)
                    violations = validate_revised(result['html_content'], contract)
                    save_json(contract_dir / "violations.json", violations)
                    progress_tracker.add_timing_info(model_name, app_name, f"Non-regression valid={violations.get('valid', False)}")
                except Exception as e:
                    progress_tracker.add_timing_info(model_name, app_name, f"Non-regression validation error: {e}")
            
        return batch_result
        
    except Exception as e:
        progress_tracker.update_status(model_name, "BATCH", f"❌ Failed: {str(e)}", error_detail=str(e))
        return {
            'success': False,
            'model': model_name,
            'error': str(e),
            'processed_apps': 0,
            'app_results': []
        }

async def _load_app_data(model_name: str, app_name: str, revision_type: str, v0_dir: str, progress_tracker, source_cua_model: str = "uitars") -> dict:
    """Load app data (initial website, failed tasks, etc.) for batch processing"""
    try:
        # Load initial website and context paths
        if v0_dir:
            v0_website_path = THIS_DIR / "initial" / v0_dir / "websites" / app_name / model_name / "index.html"
            cua_results_path = THIS_DIR / "initial" / v0_dir / "tasks" / app_name / "initial_cua_results" / model_name / source_cua_model / "results.json"
            judge_results_path = THIS_DIR / "initial" / v0_dir / "tasks" / app_name / "states" / model_name / "rules.json"
        else:
            v0_website_path = Path(f"websites/{app_name}/{model_name}/index.html")
            cua_results_path = Path(f"tasks/{app_name}/initial_cua_results/{model_name}/{source_cua_model}/results.json")
            judge_results_path = Path(f"tasks/{app_name}/states/{model_name}/rules.json")
        
        # Load initial website HTML
        if not v0_website_path.exists():
            if progress_tracker:
                progress_tracker.update_status(model_name, app_name, "❌ Initial website not found", error_detail=f"{v0_website_path}")
            return None
            
        with open(v0_website_path, 'r', encoding='utf-8') as f:
            v0_html = f.read()
        
        # Load context based on revision type
        failed_tasks = []
        unsupported_tasks = []
        
        if revision_type in ['cua', 'cua-text-only', 'cua-screenshot-only', 'integrated']:
            if cua_results_path.exists():
                with open(cua_results_path, 'r', encoding='utf-8') as f:
                    cua_results = json.load(f)
                    # Extract failed tasks from task_results where completed=false
                    for task_result in cua_results.get('task_results', []):
                        if not task_result.get('completed', False):
                            t_index = task_result.get('task_index', 0)
                            # Derive trajectory directory for this failed task (for storyboard generation)
                            if v0_dir:
                                trajectory_dir = (THIS_DIR / "initial" / v0_dir / "tasks" / app_name /
                                                  "initial_cua_results" / model_name / source_cua_model /
                                                  "trajectories" / f"task_{t_index}")
                            else:
                                trajectory_dir = (Path(f"tasks/{app_name}/initial_cua_results/{model_name}/"
                                                       f"{source_cua_model}/trajectories/task_{t_index}"))
                            failed_tasks.append({
                                'task_index': t_index,
                                'description': task_result.get('task_description', ''),
                                'steps': task_result.get('steps', 0),
                                'trajectory_dir': str(trajectory_dir)
                            })
        
        if revision_type in ['unsupported', 'integrated']:
            if judge_results_path.exists():
                with open(judge_results_path, 'r', encoding='utf-8') as f:
                    judge_results = json.load(f)
                    # rules.json stores unsupported tasks under analysis
                    unsupported_tasks = (judge_results.get('analysis', {}) or {}).get('unsupported_tasks', [])
        
        return {
            'app_name': app_name,
            'v0_html': v0_html,
            'failed_tasks': failed_tasks,
            'unsupported_tasks': unsupported_tasks
        }
        
    except Exception as e:
        if progress_tracker:
            # Surface load failure at per-app cell and timing under BATCH
            progress_tracker.update_status(model_name, app_name, f"❌ Load failed", error_detail=str(e))
            progress_tracker.add_timing_info(model_name, "BATCH", f"{app_name}: Failed to load app data: {e}")
        return None

async def _process_single_app(model_name: str, app_name: str, revision_component, progress_tracker,
                             experiment_name: str, revision_type: str, destylized: bool, 
                             v0_dir: str, variant_name: str, source_cua_model: str = "uitars") -> dict:
    """Process single app within model batch"""
    try:
        progress_tracker.update_status(model_name, app_name, "Loading initial website...")
        
        # Load initial website and context data
        if v0_dir:
            v0_website_path = THIS_DIR / "initial" / v0_dir / "websites" / app_name / model_name / "index.html"
            cua_results_path = THIS_DIR / "initial" / v0_dir / "tasks" / app_name / "initial_cua_results" / model_name / source_cua_model / "results.json"
            judge_results_path = THIS_DIR / "initial" / v0_dir / "tasks" / app_name / "states" / model_name / "rules.json"
        else:
            v0_website_path = Path(f"websites/{app_name}/{model_name}/index.html")
            cua_results_path = Path(f"tasks/{app_name}/initial_cua_results/{model_name}/{source_cua_model}/results.json")
            judge_results_path = Path(f"tasks/{app_name}/states/{model_name}/rules.json")
            
        if not v0_website_path.exists():
            return {
                'success': False,
                'error': f"Initial website not found: {v0_website_path}",
                'model': model_name,
                'app': app_name
            }
        
        with open(v0_website_path, 'r', encoding='utf-8') as f:
            v0_html = f.read()
        
        progress_tracker.update_status(model_name, app_name, "Loading CUA results...")
        
        # Load context data based on revision type
        context = {}
        if revision_type in ['cua', 'cua-text-only', 'cua-screenshot-only', 'integrated']:
            # For CUA-based revisions, failed tasks are optional; if results absent, run destylization-only
            failed_tasks = []
            if cua_results_path.exists():
                with open(cua_results_path, 'r', encoding='utf-8') as f:
                    cua_results = json.load(f)
                for task_result in cua_results.get('task_results', []):
                    if not task_result.get('completed', False):
                        failed_tasks.append({
                            'task_index': task_result.get('task_index', 0),
                            'description': task_result.get('task_description', ''),
                            'steps': task_result.get('steps', 0)
                        })
            context['failed_tasks'] = failed_tasks
        
        if revision_type in ['unsupported', 'integrated']:
            # Context for unsupported revision will be loaded by the component
            pass
        
        # Build non-regression contract
        try:
            contract = extract_contract(Path(judge_results_path), v0_html)
            contract_prompt = format_contract_prompt(contract)
        except Exception:
            contract, contract_prompt = ({}, "")

        progress_tracker.update_status(model_name, app_name, f"Starting {revision_type} revision...")
        
        # Execute revision using component
        result = await revision_component.revise(
            model_name=model_name,
            app_name=app_name,
            v0_html=v0_html,
            context={**context, 'non_regression_contract_prompt': contract_prompt},
            destylized=destylized,
            v0_dir=v0_dir,
            progress_tracker=progress_tracker
        )
        
        if result['success']:
            progress_tracker.update_status(model_name, app_name, "Saving revised website...")
            
            # Update progress with retry details if available
            if 'retry_details' in result:
                progress_tracker.update_status(
                    model_name, app_name, "✅ Generated revised website", 
                    retry_info=result['retry_details']
                )
                # 也写入timing日志，保持在屏幕上
                details = result['retry_details']
                if isinstance(details, list):
                    for at in details:
                        attempt = at.get('attempt', '?')
                        success = at.get('success', False)
                        gen_time = at.get('generation_time', None)
                        html_len = at.get('html_length', None)
                        icon = '✅' if success else '❌'
                        parts = [f"{app_name} Attempt {attempt}"]
                        if gen_time is not None:
                            parts.append(f"{gen_time}s")
                        if html_len is not None:
                            parts.append(f"{html_len} chars")
                        progress_tracker.add_timing_info(model_name, app_name, f"RETRY {icon}: {' '.join(parts)}")
                else:
                    progress_tracker.add_timing_info(model_name, app_name, f"RETRY: {app_name} {str(details)}")
            
            # Save revised website in new directory structure
            meta = {
                'model': model_name,
                'app': app_name,
                'variant': variant_name,
                'v0_signature': result.get('v0_signature'),
                'analyzed_failures': result.get('analyzed_failures'),
                'destylized': result.get('destylized'),
                'generated_at': __import__('datetime').datetime.now().isoformat()
            }
            revised_website_path = _save_revised_website(
                result['html_content'], app_name, model_name, experiment_name, variant_name, meta
            )
            # Save contract and validate revised
            try:
                contract_dir = Path(f"experiments/{experiment_name}/evaluations/{variant_name}/non_regression/{app_name}/{model_name}")
                save_json(contract_dir / "contract.json", contract)
                violations = validate_revised(result['html_content'], contract)
                save_json(contract_dir / "violations.json", violations)
                progress_tracker.add_timing_info(model_name, app_name, f"Non-regression valid={violations.get('valid', False)}")
            except Exception as e:
                progress_tracker.add_timing_info(model_name, app_name, f"Non-regression validation error: {e}")
            
            return {
                'success': True,
                'revised_website_path': revised_website_path,
                'revision_type': revision_type,
                'variant_name': variant_name,
                'destylized': destylized,
                'model': model_name,
                'app': app_name,
                **{k: v for k, v in result.items() if k not in ['success', 'html_content']}
            }
        else:
            # Update progress with retry details for failed cases
            if 'retry_details' in result:
                progress_tracker.update_status(
                    model_name, app_name, f"❌ Failed after {result.get('attempts', '?')} attempts", 
                    retry_info=result['retry_details']
                )
                # 也写入timing日志
                details = result['retry_details']
                if isinstance(details, list):
                    for at in details:
                        attempt = at.get('attempt', '?')
                        success = at.get('success', False)
                        gen_time = at.get('generation_time', None)
                        html_len = at.get('html_length', None)
                        icon = '✅' if success else '❌'
                        parts = [f"{app_name} Attempt {attempt}"]
                        if gen_time is not None:
                            parts.append(f"{gen_time}s")
                        if html_len is not None:
                            parts.append(f"{html_len} chars")
                        progress_tracker.add_timing_info(model_name, app_name, f"RETRY {icon}: {' '.join(parts)}")
                else:
                    progress_tracker.add_timing_info(model_name, app_name, f"RETRY: {app_name} {str(details)}")
            # 持久化错误到timing
            progress_tracker.add_timing_info(model_name, app_name, f"ERROR: {result.get('error', 'Unknown error')}")
            
            return {
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'revision_type': revision_type,
                'variant_name': variant_name,
                'model': model_name,
                'app': app_name
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'revision_type': revision_type,
            'variant_name': variant_name,
            'model': model_name,
            'app': app_name
        }

async def run_model_batches(runner, model_app_groups, stage_name: str, **kwargs):
    """Run model batches while displaying per-app grid (unified with other stages)."""
    from utils.progress_tracker import ProgressTracker
    
    # Models and union of all apps for display grid
    all_models = list(model_app_groups.keys())
    all_apps_set = set()
    for apps in model_app_groups.values():
        all_apps_set.update(apps)
    # Include special "BATCH" key to create one task per model, but it will be hidden from the grid
    all_apps = sorted(all_apps_set) + ["BATCH"]
    
    # Run one task per model (BATCH) but show per-app columns in ProgressTracker
    return await runner.run_parallel_tasks(
        models=all_models,
        apps=all_apps,
        task_func=revise_model_batch_task,
        stage_name=stage_name,
        valid_combinations=[(model, "BATCH") for model in all_models],
        model_app_groups=model_app_groups,
        **kwargs
    )

# Legacy function - not used in new batch approach
async def run_valid_combinations_old(runner, valid_combinations, task_func, stage_name, **kwargs):
    """运行有效的模型-应用组合"""
    from utils.progress_tracker import ProgressTracker
    
    # 创建模型和应用列表用于进度显示
    all_models = sorted(set(combo[0] for combo in valid_combinations))
    all_apps = sorted(set(combo[1] for combo in valid_combinations))
    
    # Use ParallelRunner's public interface
    return await runner.run_parallel_tasks(
        models=all_models,
        apps=all_apps, 
        task_func=task_func,
        stage_name=stage_name,
        valid_combinations=valid_combinations,
        **kwargs
    )

async def run_valid_combinations_with_tracker(runner, valid_combinations, task_func, progress_tracker, **kwargs):
    """运行有效的模型-应用组合 - 使用外部提供的progress_tracker"""
    # Create tasks for valid combinations
    async def create_task(model_name, app_name):
        async with runner.semaphore:
            return await task_func(model_name, app_name, progress_tracker, **kwargs)
    
    tasks = []
    for model_name, app_name in valid_combinations:
        task = asyncio.create_task(create_task(model_name, app_name))
        tasks.append((model_name, app_name, task))
    
    # Collect results
    results = []
    for model_name, app_name, task in tasks:
        try:
            result = await task
            results.append({
                'model': model_name,
                'app': app_name,
                'result': result
            })
        except Exception as e:
            progress_tracker.mark_failed(model_name, app_name, str(e))
            results.append({
                'model': model_name, 
                'app': app_name,
                'result': {'success': False, 'error': str(e)}
            })
    
    # Generate summary
    successful_tasks = sum(1 for r in results if r['result'].get('success', False))
    failed_tasks = len(results) - successful_tasks
    
    return {
        'total_tasks': len(results),
        'successful_tasks': successful_tasks,
        'failed_tasks': failed_tasks,
        'results': results
    }

def _generate_variant_name(revision_type: str, destylized: bool) -> str:
    """Keep base revision type for caches; destylized suffix only for non-CUA types."""
    if revision_type in ['cua', 'integrated']:
        return revision_type
    return f"{revision_type}_destylized" if destylized else revision_type

def _save_revised_website(html_content: str, app_name: str, model_name: str,
                          experiment_name: str, run_key: str, meta: dict = None) -> str:
    """Save revised website under runs/[run_key]/stage3_0 (uses revised_website path).

    Clean the entire app/model leaf to avoid stale artifacts (e.g., storyboard),
    then recreate revised_website/ and write index.html (+ optional meta.json).
    """
    from shutil import rmtree
    website_root = Path(f"experiments/{experiment_name}/runs/{run_key}/stage3_0/{app_name}/{model_name}")
    if website_root.exists():
        rmtree(website_root)
    website_dir = website_root / "revised_website"
    website_dir.mkdir(parents=True, exist_ok=True)
    website_path = website_dir / "index.html"
    with open(website_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    if meta:
        meta_path = website_dir / 'meta.json'
        with open(meta_path, 'w', encoding='utf-8') as mf:
            json.dump(meta, mf, indent=2, ensure_ascii=False)
    return str(website_path)

async def main():
    parser = argparse.ArgumentParser(description='Revise websites based on failures')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment name (e.g., exp1)')
    parser.add_argument('--models', type=str, required=True,
                       help='Comma-separated list of models (e.g., gpt5,qwen,gpt4o)')
    parser.add_argument('--apps', type=str, required=True,
                       help='Comma-separated list of apps or "all" for all 52 apps')
    parser.add_argument('--revision-type', type=str, default='cua',
                       choices=['cua', 'unsupported', 'integrated'],
                       help='Revision type: cua, unsupported, integrated')
    parser.add_argument('--commenter', type=str, default='none',
                       choices=['none', 'cua-text-only', 'cua-screenshot-only', 'full'],
                       help='Commenter variant for CUA-based revisions')
    parser.add_argument('--destylized', type=str, default='auto',
                       help="Derived from revision-type; 'auto' is ignored")
    parser.add_argument('--max-concurrent', type=int, default=5,
                       help='Maximum concurrent tasks')
    parser.add_argument('--commenter-concurrent', type=int, default=10,
                       help='Maximum concurrent commenter calls')
    parser.add_argument('--initial-dir', type=str, default=None,
                       help='Initial data directory name (stored under initial/[dir])')
    parser.add_argument('--source-cua-model', type=str, default='uitars',
                       help='CUA model whose initial run failures are used as revision input (default: uitars)')
    parser.add_argument('--force-comments', action='store_true',
                       help='Force re-generate commenter analyses (ignore cached results)')
    parser.add_argument('--force-revised', action='store_true',
                       help='Force re-generate revised websites (ignore cached revised)')
    
    args = parser.parse_args()

    # Parse destylized preference with default true for CUA/integrated,
    # and allow explicit override via '--destylized=false'
    # Destylization is tied to revision type (CUA/integrated on; unsupported off)
    args.destylized = True if args.revision_type in ['cua', 'integrated'] else False
    
    # 解析模型列表
    models = args.models.split(',')
    
    # 解析应用列表
    if args.apps.lower() == 'all':
        apps = DEFAULT_APPS
    else:
        apps = args.apps.split(',')
    
    # Generate variant name
    variant_name = _generate_variant_name(args.revision_type, args.destylized)
    
    ts_print(f"Stage 3: Revising websites - {args.revision_type} revision")
    ts_print(f"Experiment: {args.experiment}")
    ts_print(f"Variant: {variant_name}")
    ts_print(f"Models: {models}")
    ts_print(f"Apps: {apps}")
    if args.destylized:
        ts_print(f"Destylization: Enabled (simple color scheme, fit screen)")

    # Validate and filter model-app combinations based on revision type
    valid_combinations = []
    skipped_combinations = []
    
    for app in apps:
        for model in models:
            if args.initial_dir:
                v0_website_path = Path(f"initial/{args.initial_dir}/websites/{app}/{model}/index.html")
                cua_results_path = Path(f"initial/{args.initial_dir}/tasks/{app}/initial_cua_results/{model}/{args.source_cua_model}/results.json")
                judge_results_path = Path(f"initial/{args.initial_dir}/tasks/{app}/states/{model}/rules.json")
            else:
                v0_website_path = Path(f"websites/{app}/{model}/index.html")
                cua_results_path = Path(f"tasks/{app}/initial_cua_results/{model}/{args.source_cua_model}/results.json")
                judge_results_path = Path(f"tasks/{app}/states/{model}/rules.json")
                
            missing_files = []
            
            # Always need initial website
            if not v0_website_path.exists():
                missing_files.append(f"Initial website: {v0_website_path}")
            
            # Check revision-type specific requirements
            # For CUA-based revisions, allow running even without initial CUA results;
            # destylization-only path will still produce a revised site.
            # Therefore, do NOT require CUA results for 'cua', 'cua-text-only',
            # 'cua-screenshot-only', or 'integrated'.
            
            if args.revision_type in ['unsupported', 'integrated']:
                if not judge_results_path.exists():
                    missing_files.append(f"Judge results: {judge_results_path}")
            
            if not missing_files:
                valid_combinations.append((model, app))
            else:
                skipped_combinations.append((model, app, missing_files))
    
    if not valid_combinations:
        ts_print("❌ No valid model-app combinations found with all required files")
        return
    
    if skipped_combinations:
        ts_print("⚠️  Skipping combinations with missing files:")
        for model, app, missing_files in skipped_combinations:
            ts_print(f"  {model}/{app}: {', '.join(missing_files)}")
        ts_print("")
    
    ts_print(f"✅ Processing {len(valid_combinations)} valid combinations")
    
    # Create experiment directory structure
    exp_dir = REPO_ROOT / "experiments" / args.experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summaries directory
    summaries_dir = exp_dir / "summaries" / "stage3_0_revise"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment configuration
    exp_config = {
        'experiment_name': args.experiment,
        'stage': 'Stage 3: Website Revision',
        'revision_type': args.revision_type,
        'variant_name': variant_name,
        'destylized': args.destylized,
        'requested_models': models,
        'requested_apps': apps,
        'valid_combinations': valid_combinations,
        'skipped_combinations': [(m, a) for m, a, _ in skipped_combinations],
        'description': f'Generate revised websites using {args.revision_type} revision'
    }
    
    with open(exp_dir / f"config_{variant_name}.json", 'w', encoding='utf-8') as f:
        json.dump(exp_config, f, indent=2, ensure_ascii=False)
    
    # Group apps by model for batch processing
    model_app_groups = {}
    for model, app in valid_combinations:
        if model not in model_app_groups:
            model_app_groups[model] = []
        model_app_groups[model].append(app)
    
    # Build run key for this configuration
    run_key = build_run_key(args.revision_type, args.commenter, args.initial_dir)
    rk_short = short_run_key(run_key)

    ts_print(f"🔄 Batch processing: {list(model_app_groups.keys())} models")
    for model, apps in model_app_groups.items():
        ts_print(f"  {model}: {len(apps)} apps")
    
    # Use ParallelRunner for model-level parallelism (not model-app combinations)
    runner = ParallelRunner(
        max_concurrent=len(model_app_groups),  # One task per model
        api_max_concurrent=len(model_app_groups), 
        local_max_concurrent=len(model_app_groups)
    )
    
    summary = await run_model_batches(
        runner, model_app_groups,
        stage_name=f"[{rk_short}] Stage 3: {args.revision_type} ({args.commenter})",
        experiment_name=args.experiment,
        revision_type=args.revision_type,
        destylized=args.destylized,
        v0_dir=args.initial_dir,
        commenter_concurrent=args.commenter_concurrent,
        force_comments=args.force_comments,
        force_v1=args.force_revised,
        source_cua_model=args.source_cua_model,
        commenter=args.commenter,
        run_key=run_key
    )
    
    # Calculate revision statistics
    successful_revisions = 0
    revision_stats = {
        'destylized': 0,
        'total_versions_generated': 0
    }
    
    for result in summary['results']:
        if result['result'].get('success'):
            successful_revisions += 1
            # Track generation details if available
            generation_details = result['result'].get('generation_details', [])
            revision_stats['total_versions_generated'] += len(generation_details)
            if result['result'].get('destylized', False):
                revision_stats['destylized'] += 1
    
    # Create detailed summary
    detailed_summary = {
        **summary,
        'experiment_name': args.experiment,
        'revision_type': args.revision_type,
        'variant_name': variant_name,
        'destylized': args.destylized,
        'successful_revisions': successful_revisions,
        'revision_success_rate': successful_revisions / summary['total_tasks'] if summary['total_tasks'] > 0 else 0,
        'revision_stats': revision_stats,
        'requested_models': models,
        'requested_apps': apps,
        'valid_combinations': valid_combinations,
        'skipped_combinations': [(m, a) for m, a, _ in skipped_combinations] if 'skipped_combinations' in locals() else []
    }
    
    # Save to experiments progress with variant name
    summary_path = THIS_DIR / "progress" / "experiments" / args.experiment / "summaries" / "stage3_0_revise" / f"{run_key}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_summary, f, indent=2, ensure_ascii=False)
    
    # Save variant-specific summary in experiment summaries directory
    variant_summary_path = summaries_dir / f"{run_key}_summary.json"
    with open(variant_summary_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_summary, f, indent=2, ensure_ascii=False)
    
    ts_print(f"\n📊 {variant_name.title()} Revision Summary:")
    ts_print(f"✅ Successful revisions: {summary['successful_tasks']}/{summary['total_tasks']}")
    if revision_stats['total_versions_generated'] > 0:
        ts_print(f"📊 Total versions generated: {revision_stats['total_versions_generated']}")
    if args.destylized and revision_stats['destylized'] > 0:
        ts_print(f"🎨 Destylized: {revision_stats['destylized']} revisions")
    ts_print(f"📁 Summary saved to: {summary_path}")
    ts_print(f"📁 Variant summary saved to: {variant_summary_path}")
    
    if summary['failed_tasks'] > 0:
        ts_print("\n❌ Failed revisions:")
        for result in summary['results']:
            if not result['result'].get('success'):
                ts_print(f"  {result['model']}/{result['app']}: {result['result'].get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(main())
