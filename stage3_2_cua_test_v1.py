#!/usr/bin/env python3
"""
Stage 3.2: CUA test on revised websites.
Use UI‑TARS 7B to execute tasks on revised websites.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.model_client import ModelClient
from utils.parallel_runner import ParallelRunner
from utils.run_key import build_run_key, short_run_key
from agents.cua_policy import create_cua_policy
from utils.constants import DEFAULT_APPS


async def cua_test_v1_task(model_name: str, app_name: str, progress_tracker, experiment_name: str = "exp1", run_key: str = None, v0_dir: str = None, cua_model: str = "uitars", **kwargs) -> dict:
    """Single CUA test task for one revised website."""
    try:
        model_client = ModelClient()
        cua_policy = create_cua_policy(model_client, cua_model_name=cua_model, max_steps=20)
        
        progress_tracker.update_status(model_name, app_name, "Loading revised rules...")
        
        # Load v1 rules (run_key layout)
        v1_rules_path = Path(f"experiments/{experiment_name}/runs/{run_key}/stage3_1/{app_name}/{model_name}/rules.json")
        
        if not v1_rules_path.exists():
            return {
                'success': False,
                'error': f"Revised rules not found: {v1_rules_path}",
                'model': model_name,
                'app': app_name
            }
        
        with open(v1_rules_path, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
        
        # Get supported tasks
        supported_tasks = rules_data.get('analysis', {}).get('supported_tasks', [])
        
        if not supported_tasks:
            return {
                'success': True,
                'completed_tasks': 0,
                'total_tasks': 0,
                'results': [],
                'message': "No supported tasks found in revised",
                'model': model_name,
                'app': app_name
            }
        
        progress_tracker.update_status(model_name, app_name, f"Testing {len(supported_tasks)} revised tasks...")
        
        # Build revised website URL (path uses revised_website)
        revised_website_path = Path(f"experiments/{experiment_name}/runs/{run_key}/stage3_0/{app_name}/{model_name}/revised_website/index.html").absolute()
        
        website_url = f"file://{revised_website_path}"
        
        # Execute tasks
        task_results = []
        completed_count = 0
        
        # Load task description mapping
        if v0_dir:
            tasks_file = f"initial/{v0_dir}/tasks/{app_name}/tasks.json"
        else:
            tasks_file = f"tasks/{app_name}/tasks.json"
        with open(tasks_file) as f:
            tasks_data = json.load(f)
        task_map = {t['id']: t['description'] for t in tasks_data['tasks']}
        
        # Clean and create directory for saving trajectories and results
        from shutil import rmtree
        base_out_dir = Path(f"experiments/{experiment_name}/runs/{run_key}/stage3_2/{cua_model}/{app_name}/{model_name}")
        if base_out_dir.exists():
            rmtree(base_out_dir)
        trajectories_dir = base_out_dir / "trajectories"
        trajectories_dir.mkdir(parents=True, exist_ok=True)
        results_jsonl = base_out_dir / "results.jsonl"
        
        for i, task_info in enumerate(supported_tasks):
            task_id = task_info['task_index']
            task_description = task_map[task_id]
            completion_rule = task_info.get('rule', '')
            
            progress_tracker.update_status(
                model_name, app_name,
                f"[{task_id}] Revised Task {i+1}/{len(supported_tasks)}: {task_description[:30]}..."
            )
            
            # Create a separate trajectory directory per task (keyed by task_id)
            task_trajectory_dir = trajectories_dir / f"{task_id}"
            task_trajectory_dir.mkdir(parents=True, exist_ok=True)
            
            # Execute task
            _t_start = datetime.now().isoformat()
            result = await cua_policy.execute_task(
                app_name=app_name,
                model_name=model_name,
                website_url=website_url,
                task={'description': task_description},
                completion_rule=completion_rule,
                save_dir=str(task_trajectory_dir)
            )
            _t_end = datetime.now().isoformat()
            
            # Save trajectory.jsonl (one line per step)
            traj_jsonl = task_trajectory_dir / "trajectory.jsonl"
            if 'trajectory' in result:
                with open(traj_jsonl, 'w', encoding='utf-8') as tf:
                    for step_item in result['trajectory']:
                        tf.write(json.dumps(step_item, ensure_ascii=False) + "\n")

            # Append per-task result to results.jsonl
            rec = {
                'run_key': run_key,
                'cua_model': cua_model,
                'model': model_name,
                'app': app_name,
                'task_id': task_id,
                'task_description': task_description,
                'action_steps': result.get('steps', 0),
                'success': bool(result.get('success', False)),
                'completed': bool(result.get('completed', False)),
                'state_diffs': None,
                'reason': result.get('termination_reason', ''),
                'trajectory_dir': str(task_trajectory_dir),
                'timestamps': {'start': _t_start, 'end': _t_end}
            }
            with open(results_jsonl, 'a', encoding='utf-8') as rf:
                rf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            task_results.append(rec)
            
            if result.get('completed', False):
                completed_count += 1
        
        # Load initial run results for comparison
        if v0_dir:
            v0_results_path = Path(f"initial/{v0_dir}/tasks/{app_name}/initial_cua_results/{model_name}/{cua_model}/results.json")
        else:
            v0_results_path = Path(f"tasks/{app_name}/initial_cua_results/{model_name}/{cua_model}/results.json")
        v0_completed = 0
        v0_total = 0
        
        if v0_results_path.exists():
            with open(v0_results_path, 'r', encoding='utf-8') as f:
                v0_results = json.load(f)
            v0_completed = v0_results.get('completed_tasks', 0)
            v0_total = v0_results.get('tested_tasks', 0)
        
        # Save aggregated overview (overwrite)
        summary = {
            'run_key': run_key,
            'cua_model': cua_model,
            'app_name': app_name,
            'model_name': model_name,
            'total_supported_tasks': len(supported_tasks),
            'tested_tasks': len(task_results),
            'completed_tasks': completed_count,
            'success_rate': completed_count / len(task_results) if task_results else 0,
            'v0_comparison': {
                'v0_completed': v0_completed,
                'v0_total': v0_total,
                'v1_completed': completed_count,
                'v1_total': len(task_results),
                'improvement': completed_count - v0_completed
            }
        }
        with open(base_out_dir / "run_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return {
            'success': True,
            'v0_completed': v0_completed,
            'v1_completed': completed_count,
            'improvement': completed_count - v0_completed,
            'total_tasks': len(task_results),
            'success_rate': completed_count / len(task_results) if task_results else 0,
            'results_file': str(results_jsonl),
            'model': model_name,
            'app': app_name
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'model': model_name,
            'app': app_name
        }

async def main():
    parser = argparse.ArgumentParser(description='CUA test revised websites')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment name (e.g., exp1)')
    parser.add_argument('--models', type=str, required=True,
                       help='Comma-separated list of models (e.g., gpt5,qwen,gpt4o)')
    parser.add_argument('--apps', type=str, required=True,
                       help='Comma-separated list of apps or "all" for all 52 apps')
    parser.add_argument('--revision-type', type=str, default='cua',
                       help='Revision type to test (cua, unsupported, integrated)')
    parser.add_argument('--commenter', type=str, default='none',
                       choices=['none', 'cua-text-only', 'cua-screenshot-only', 'full'],
                       help='Commenter variant to reconstruct run_key')
    parser.add_argument('--max-concurrent', type=int, default=20,
                       help='Maximum concurrent tasks (limited due to browser resources)')
    parser.add_argument('--initial-dir', type=str, default=None,
                       help='Initial data directory name (stored under initial/[dir])')
    parser.add_argument('--cua-models', type=str, default='uitars',
                       help='Comma-separated list of CUA models (e.g., uitars,operator)')
    
    args = parser.parse_args()
    
    # Parse model list
    models = args.models.split(',')
    
    # Parse app list
    if args.apps.lower() == 'all':
        apps = DEFAULT_APPS
    else:
        apps = args.apps.split(',')
    
    # Build run_key
    revision_type = args.revision_type
    run_key = build_run_key(revision_type, args.commenter, args.initial_dir)
    rk_short = short_run_key(run_key)
    
    # Check experiment directory structure
    exp_dir = Path(f"experiments/{args.experiment}")
    if not exp_dir.exists():
        print(f"Experiment directory not found: {exp_dir}")
        return
    
    # Gating by presence of required inputs per-combination (checked below)
    
    print(f"Stage 3.2: CUA testing revised websites")
    print(f"Experiment: {args.experiment}")
    print(f"Run key: {run_key}")
    print(f"Models: {models}")
    print(f"Apps: {apps}")
    # Parse CUA models list
    cua_models = args.cua_models.split(',')
    print(f"CUA models: {cua_models}")
    
    # Check required files and filter valid combinations
    valid_combinations = []
    skipped_combinations = []
    
    for model in models:
        for app in apps:
            # Check run_key paths
            revised_website_path = Path(f"experiments/{args.experiment}/runs/{run_key}/stage3_0/{app}/{model}/revised_website/index.html")
            v1_rules_path = Path(f"experiments/{args.experiment}/runs/{run_key}/stage3_1/{app}/{model}/rules.json")
            
            if revised_website_path.exists() and v1_rules_path.exists():
                valid_combinations.append((model, app))
            else:
                missing_files = []
                if not revised_website_path.exists():
                    missing_files.append(f"Revised website: {revised_website_path}")
                if not v1_rules_path.exists():
                    missing_files.append(f"Revised rules: {v1_rules_path}")
                skipped_combinations.append((model, app, missing_files))
    
    if not valid_combinations:
        print("❌ No valid model-app combinations found with all required files")
        return
    
    if skipped_combinations:
        print("⚠️ Skipping combinations with missing files:")
        for model, app, missing_files in skipped_combinations:
            print(f"  {model}/{app}: {', '.join(missing_files)}")
        print()
    
    # Update model and app lists to valid combinations only
    valid_models = list(set(combo[0] for combo in valid_combinations))
    valid_apps = list(set(combo[1] for combo in valid_combinations))
    print(f"✅ Processing {len(valid_combinations)} valid combinations")
    print(f"Models: {valid_models}")
    print(f"Apps: {valid_apps}")
    print()
    
    # Create parallel runner
    runner = ParallelRunner(max_concurrent=args.max_concurrent)
    
    # Run tasks for each CUA model
    all_results = []
    for cua_model in cua_models:
        print(f"\n🤖 Testing with CUA model: {cua_model}")
        summary = await runner.run_parallel_tasks(
            models=valid_models,
            apps=valid_apps,
            task_func=cua_test_v1_task,
            stage_name=f"[{rk_short}] Stage 3.2: {cua_model}",
            experiment_name=args.experiment,
            run_key=run_key,
            valid_combinations=valid_combinations,
            v0_dir=args.initial_dir,
            cua_model=cua_model
        )
        all_results.append({
            'cua_model': cua_model,
            'summary': summary
        })
    
    # Aggregate statistics across all CUA models
    total_initial_completed = 0
    total_revised_completed = 0
    total_improvement = 0
    total_tested = 0
    
    for cua_result in all_results:
        for result in cua_result['summary']['results']:
            if result['result'].get('success'):
                result_data = result['result']
                total_initial_completed += result_data.get('v0_completed', 0)
                total_revised_completed += result_data.get('v1_completed', 0)
                total_improvement += result_data.get('improvement', 0)
                total_tested += result_data.get('total_tasks', 0)
    
    # Save summary (including all CUA models)
    detailed_summary = {
        'cua_models': cua_models,
        'all_results': all_results,
        'experiment_name': args.experiment,
        'revision_type': revision_type,
        'total_initial_completed': total_initial_completed,
        'total_revised_completed': total_revised_completed,
        'total_improvement': total_improvement,
        'total_tested_tasks': total_tested,
        'initial_success_rate': total_initial_completed / total_tested if total_tested > 0 else 0,
        'revised_success_rate': total_revised_completed / total_tested if total_tested > 0 else 0
    }
    
    # Save local summary
    local_summary_path = Path(f"experiments/{args.experiment}/summaries/stage3_2_cua_test_v1/{run_key}_summary.json")
    
    local_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_summary_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_summary, f, indent=2, ensure_ascii=False)
    
    # Save global summary
    base = Path(__file__).resolve().parent
    summary_path = base / "progress" / "experiments" / args.experiment / "summaries" / "stage3_2_cua_test_v1" / f"{run_key}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_summary, f, indent=2, ensure_ascii=False)
    
    # Generate detailed evaluation output
    eval_dir = base / "progress" / "experiments" / args.experiment / "evaluations" / run_key / "stage3_2_cua_test_v1"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Model comparison analysis
    model_stats = {}
    app_stats = {}
    
    for cua_result in all_results:
        cua_model = cua_result['cua_model']
        for result in cua_result['summary']['results']:
            if result['result'].get('success'):
                model = result['model']
                app = result['app']
                v0_completed = result['result'].get('v0_completed', 0)
                v1_completed = result['result'].get('v1_completed', 0)
                improvement = result['result'].get('improvement', 0)
                total = result['result'].get('total_tasks', 0)
                success_rate = result['result'].get('success_rate', 0)
                
                # Create combined key for model+cua_model
                model_key = f"{model}+{cua_model}"
                
                if model_key not in model_stats:
                    model_stats[model_key] = {
                        'initial_completed': 0, 'revised_completed': 0, 'improvement': 0, 'total': 0, 'apps': []
                    }
                model_stats[model_key]['initial_completed'] += v0_completed
                model_stats[model_key]['revised_completed'] += v1_completed
                model_stats[model_key]['improvement'] += improvement
                model_stats[model_key]['total'] += total
                model_stats[model_key]['apps'].append({
                    'app': app,
                    'initial_completed': v0_completed,
                    'revised_completed': v1_completed,
                    'improvement': improvement,
                    'total': total,
                    'initial_success_rate': v0_completed / total if total > 0 else 0,
                    'revised_success_rate': v1_completed / total if total > 0 else 0
                })
                
                if app not in app_stats:
                    app_stats[app] = {'models': []}
                app_stats[app]['models'].append({
                    'model': model,
                    'cua_model': cua_model,
                    'initial_completed': v0_completed,
                    'revised_completed': v1_completed,
                    'improvement': improvement,
                    'total': total,
                    'initial_success_rate': v0_completed / total if total > 0 else 0,
                    'revised_success_rate': v1_completed / total if total > 0 else 0
                })
    
    # Save model comparison
    model_comparison = {
        'overview': {
            model_key: {
                'total_initial_completed': stats['initial_completed'],
                'total_revised_completed': stats['revised_completed'],
                'total_improvement': stats['improvement'],
                'total_tested': stats['total'],
                'initial_success_rate': stats['initial_completed'] / stats['total'] if stats['total'] > 0 else 0,
                'revised_success_rate': stats['revised_completed'] / stats['total'] if stats['total'] > 0 else 0,
                'num_apps': len(stats['apps'])
            } for model_key, stats in model_stats.items()
        },
        'detailed_by_model': model_stats,
        'detailed_by_app': app_stats
    }
    
    with open(eval_dir / "model_comparison.json", 'w', encoding='utf-8') as f:
        json.dump(model_comparison, f, indent=2, ensure_ascii=False)
    
    # Save individual model detailed results
    for model_key, stats in model_stats.items():
        model_detailed = {
            'model': model_key,
            'experiment': args.experiment,
            'summary': {
                'total_v0_completed': stats['initial_completed'],
                'total_v1_completed': stats['revised_completed'],
                'total_improvement': stats['improvement'],
                'total_tested': stats['total'],
                'v0_success_rate': stats['initial_completed'] / stats['total'] if stats['total'] > 0 else 0,
                'v1_success_rate': stats['revised_completed'] / stats['total'] if stats['total'] > 0 else 0,
                'num_apps_tested': len(stats['apps'])
            },
            'apps': stats['apps']
        }
        
        with open(eval_dir / f"{model_key}_detailed.json", 'w', encoding='utf-8') as f:
            json.dump(model_detailed, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Revised CUA Test Summary (All CUA Models):")
    total_successful = sum(cua_result['summary']['successful_tasks'] for cua_result in all_results)
    total_tasks = sum(cua_result['summary']['total_tasks'] for cua_result in all_results)
    print(f"✅ Successful tests: {total_successful}/{total_tasks}")
    if total_tested > 0:
        print(f"🎯 Initial completed: {total_initial_completed}/{total_tested} ({total_initial_completed/total_tested*100:.1f}%)")
        print(f"🎯 Revised completed: {total_revised_completed}/{total_tested} ({total_revised_completed/total_tested*100:.1f}%)")
    else:
        print(f"🎯 Initial completed: {total_initial_completed}/0 (0.0%)")
        print(f"🎯 Revised completed: {total_revised_completed}/0 (0.0%)")
    print(f"📈 Improvement: +{total_improvement} tasks")
    print(f"📁 Summary saved to: {summary_path}")
    
    failed_tests = []
    for cua_result in all_results:
        for result in cua_result['summary']['results']:
            if not result['result'].get('success'):
                failed_tests.append(f"  {result['model']}/{result['app']} ({cua_result['cua_model']}): {result['result'].get('error', 'Unknown error')}")
    
    if failed_tests:
        print("\n❌ Failed tests:")
        for error in failed_tests:
            print(error)

if __name__ == "__main__":
    asyncio.run(main())
