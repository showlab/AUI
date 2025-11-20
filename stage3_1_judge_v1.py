#!/usr/bin/env python3
"""
Stage 3.1: Judge evaluation on revised websites.
Use the Judge model to analyze revised websites and re-extract state rules.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.model_client import ModelClient
from utils.parallel_runner import ParallelRunner
from utils.run_key import build_run_key, short_run_key
from agents.judge import Judge
from utils.constants import DEFAULT_APPS


async def judge_revised_website_task(model_name: str, app_name: str, progress_tracker, experiment_name: str = "exp1", run_key: str = None, v0_dir: str = None, **kwargs) -> dict:
    """Single Judge task for one revised website."""
    try:
        model_client = ModelClient()
        judge = Judge(model_client)
        
        progress_tracker.update_status(model_name, app_name, "Loading revised website...")
        
        # Build revised website path (run_key layout)
        revised_website_path = Path(f"experiments/{experiment_name}/runs/{run_key}/stage3_0/{app_name}/{model_name}/revised_website/index.html")
        
        if not revised_website_path.exists():
            return {
                'success': False,
                'error': f"Revised website not found: {revised_website_path}",
                'model': model_name,
                'app': app_name
            }
        
        with open(revised_website_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        progress_tracker.update_status(model_name, app_name, "Loading tasks...")
        
        # Load tasks (reuse the same task set)
        if v0_dir:
            tasks_path = Path(f"initial/{v0_dir}/tasks/{app_name}/tasks.json")
        else:
            tasks_path = Path(f"tasks/{app_name}/tasks.json")
        if not tasks_path.exists():
            return {
                'success': False,
                'error': f"Tasks not found: {tasks_path}",
                'model': model_name,
                'app': app_name
            }
        
        with open(tasks_path, 'r', encoding='utf-8') as f:
            tasks_data = json.load(f)
        
        tasks = tasks_data.get('tasks', [])
        
        progress_tracker.update_status(model_name, app_name, "Analyzing revised website...")
        
        # Analyze v1 website and tasks (single rules output)
        analysis_result = await judge.analyze_website_tasks(app_name, html_content, tasks)
        
        if analysis_result['success']:
            progress_tracker.update_status(model_name, app_name, "Saving revised rules...")
            
            # Save v1 rules to runs/[run_key]/stage3_1 and clean sibling directory
            from shutil import rmtree
            rules_dir = Path(f"experiments/{experiment_name}/runs/{run_key}/stage3_1/{app_name}/{model_name}")
            if rules_dir.exists():
                rmtree(rules_dir)
            rules_dir.mkdir(parents=True, exist_ok=True)
            rules_file = rules_dir / "rules.json"
            with open(rules_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, ensure_ascii=False)
            
            # Optionally compare v0 vs v1 support counts
            if v0_dir:
                v0_rules_path = Path(f"initial/{v0_dir}/tasks/{app_name}/states/{model_name}/rules.json")
            else:
                v0_rules_path = Path(f"tasks/{app_name}/states/{model_name}/rules.json")
            v0_supported_count = 0
            if v0_rules_path.exists():
                try:
                    with open(v0_rules_path, 'r', encoding='utf-8') as f:
                        v0_rules = json.load(f)
                    v0_supported_count = v0_rules.get('supported_count', 0)
                except Exception:
                    pass
            v1_supported_count = analysis_result.get('supported_count', 0)
            improvement = v1_supported_count - v0_supported_count
            
            return {
                'success': True,
                'analysis': analysis_result['analysis'],
                'initial_supported_count': v0_supported_count,
                'revised_supported_count': v1_supported_count,
                'improvement': improvement,
                'total_tasks': analysis_result['total_tasks'],
                'rules_file': str(rules_file),
                'model': model_name,
                'app': app_name
            }
        else:
            return {
                'success': False,
                'error': analysis_result['error'],
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
    parser = argparse.ArgumentParser(description='Judge evaluate revised websites')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment name (e.g., exp1)')
    parser.add_argument('--models', type=str, required=True,
                       help='Comma-separated list of models (e.g., gpt5,qwen,gpt4o)')
    parser.add_argument('--apps', type=str, required=True,
                       help='Comma-separated list of apps or "all" for all 52 apps')
    parser.add_argument('--revision-type', type=str, default='cua',
                       help='Revision type to evaluate (cua, unsupported, integrated)')
    parser.add_argument('--commenter', type=str, default='none',
                       choices=['none', 'cua-text-only', 'cua-screenshot-only', 'full'],
                       help='Commenter variant to reconstruct run_key')
    parser.add_argument('--max-concurrent', type=int, default=10,
                       help='Maximum concurrent tasks (limited due to judge model API)')
    parser.add_argument('--initial-dir', type=str, default=None,
                       help='Initial data directory name (stored under initial/[dir])')
    
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
    
    # Gating via presence of revised sites under run_key (checked later per combo)
    
    print(f"Stage 3.1: Judge evaluating revised websites")
    print(f"Experiment: {args.experiment}")
    print(f"Run key: {run_key}")
    print(f"Models: {models}")
    print(f"Apps: {apps}")
    print(f"Using GPT-5 to analyze revised websites and extract rules")
    
    # Check required files and filter valid combinations
    valid_combinations = []
    skipped_combinations = []
    
    for model in models:
        for app in apps:
            # Check run_key paths
            revised_website_path = Path(f"experiments/{args.experiment}/runs/{run_key}/stage3_0/{app}/{model}/revised_website/index.html")
            if args.initial_dir:
                tasks_path = Path(f"initial/{args.initial_dir}/tasks/{app}/tasks.json")
            else:
                tasks_path = Path(f"tasks/{app}/tasks.json")
            
            if revised_website_path.exists() and tasks_path.exists():
                valid_combinations.append((model, app))
            else:
                missing_files = []
                if not revised_website_path.exists():
                    missing_files.append(f"Revised website: {revised_website_path}")
                if not tasks_path.exists():
                    missing_files.append(f"Tasks: {tasks_path}")
                skipped_combinations.append((model, app, missing_files))
    
    if not valid_combinations:
        print("❌ No valid model-app combinations found with all required files")
        return
    
    if skipped_combinations:
        print("⚠️ Skipping combinations with missing files:")
        for model, app, missing_files in skipped_combinations:
            print(f"  {model}/{app}: {', '.join(missing_files)}")
        print()
    
    # Update model/app lists to valid combinations only
    valid_models = list(set(combo[0] for combo in valid_combinations))
    valid_apps = list(set(combo[1] for combo in valid_combinations))
    print(f"✅ Processing {len(valid_combinations)} valid combinations")
    print(f"Models: {valid_models}")
    print(f"Apps: {valid_apps}")
    print()
    
    # Create parallel runner
    runner = ParallelRunner(max_concurrent=args.max_concurrent)
    
    # Run Judge tasks
    summary = await runner.run_parallel_tasks(
        models=valid_models,
        apps=valid_apps,
        task_func=judge_revised_website_task,
        stage_name=f"[{rk_short}] Stage 3.1: Judge Revised",
        experiment_name=args.experiment,
        run_key=run_key,
        valid_combinations=valid_combinations,
        v0_dir=args.initial_dir
    )
    
    # Aggregate statistics
    total_v0_supported = 0
    total_v1_supported = 0
    total_improvement = 0
    total_tasks = 0
    
    for result in summary['results']:
        if result['result'].get('success'):
            result_data = result['result']
            total_v0_supported += result_data.get('initial_supported_count', 0)
            total_v1_supported += result_data.get('revised_supported_count', 0)
            total_improvement += result_data.get('improvement', 0)
            total_tasks += result_data.get('total_tasks', 0)
    
    # Save summary payload
    detailed_summary = {
        **summary,
        'experiment_name': args.experiment,
        'revision_type': revision_type,
        'total_initial_supported': total_v0_supported,
        'total_revised_supported': total_v1_supported,
        'total_improvement': total_improvement,
        'total_tasks_analyzed': total_tasks,
        'initial_support_rate': total_v0_supported / total_tasks if total_tasks > 0 else 0,
        'revised_support_rate': total_v1_supported / total_tasks if total_tasks > 0 else 0
    }
    
    # Save local summary
    local_summary_path = Path(f"experiments/{args.experiment}/summaries/stage3_1_judge_v1/{run_key}_summary.json")
    
    local_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_summary_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_summary, f, indent=2, ensure_ascii=False)
    
    # Save global summary
    base = Path(__file__).resolve().parent
    summary_path = base / "progress" / "experiments" / args.experiment / "summaries" / "stage3_1_judge_v1" / f"{run_key}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_summary, f, indent=2, ensure_ascii=False)
    
    # Generate detailed evaluation output
    eval_dir = base / "progress" / "experiments" / args.experiment / "evaluations" / run_key / f"stage3_1_judge_v1"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Model comparison analysis
    model_stats = {}
    app_stats = {}
    
    for result in summary['results']:
        if result['result'].get('success'):
            model = result['model']
            app = result['app']
            v0_supported = result['result'].get('initial_supported_count', 0)
            v1_supported = result['result'].get('revised_supported_count', 0)
            improvement = result['result'].get('improvement', 0)
            total = result['result'].get('total_tasks', 0)
            
            if model not in model_stats:
                model_stats[model] = {
                    'initial_supported': 0, 'revised_supported': 0, 'improvement': 0, 'total': 0, 'apps': []
                }
            model_stats[model]['initial_supported'] += v0_supported
            model_stats[model]['revised_supported'] += v1_supported
            model_stats[model]['improvement'] += improvement
            model_stats[model]['total'] += total
            model_stats[model]['apps'].append({
                'app': app,
                'initial_supported': v0_supported,
                'revised_supported': v1_supported,
                'improvement': improvement,
                'total': total,
                'initial_rate': v0_supported / total if total > 0 else 0,
                'revised_rate': v1_supported / total if total > 0 else 0
            })
            
            if app not in app_stats:
                app_stats[app] = {'models': []}
            app_stats[app]['models'].append({
                'model': model,
                'initial_supported': v0_supported,
                'revised_supported': v1_supported,
                'improvement': improvement,
                'total': total,
                'initial_rate': v0_supported / total if total > 0 else 0,
                'revised_rate': v1_supported / total if total > 0 else 0
            })
    
    # Save model comparison
    model_comparison = {
        'overview': {
            model: {
                'total_initial_supported': stats['initial_supported'],
                'total_revised_supported': stats['revised_supported'],
                'total_improvement': stats['improvement'],
                'total_tasks': stats['total'],
                'initial_support_rate': stats['initial_supported'] / stats['total'] if stats['total'] > 0 else 0,
                'revised_support_rate': stats['revised_supported'] / stats['total'] if stats['total'] > 0 else 0,
                'num_apps': len(stats['apps'])
            } for model, stats in model_stats.items()
        },
        'detailed_by_model': model_stats,
        'detailed_by_app': app_stats
    }
    
    with open(eval_dir / "model_comparison.json", 'w', encoding='utf-8') as f:
        json.dump(model_comparison, f, indent=2, ensure_ascii=False)
    
    # Save individual model detailed results
    for model, stats in model_stats.items():
        model_detailed = {
            'model': model,
            'experiment': args.experiment,
            'summary': {
                'total_initial_supported': stats['initial_supported'],
                'total_revised_supported': stats['revised_supported'],
                'total_improvement': stats['improvement'],
                'total_tasks': stats['total'],
                'initial_support_rate': stats['initial_supported'] / stats['total'] if stats['total'] > 0 else 0,
                'revised_support_rate': stats['revised_supported'] / stats['total'] if stats['total'] > 0 else 0,
                'num_apps_tested': len(stats['apps'])
            },
            'apps': stats['apps']
        }
        
        with open(eval_dir / f"{model}_detailed.json", 'w', encoding='utf-8') as f:
            json.dump(model_detailed, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Revised Analysis Summary:")
    print(f"✅ Successful analyses: {summary['successful_tasks']}/{summary['total_tasks']}")
    if total_tasks > 0:
        print(f"📈 Initial supported: {total_v0_supported}/{total_tasks} ({total_v0_supported/total_tasks*100:.1f}%)")
    else:
        print(f"📈 Initial supported: {total_v0_supported}/0 (0.0%)")
    if total_tasks > 0:
        print(f"📈 Revised supported: {total_v1_supported}/{total_tasks} ({total_v1_supported/total_tasks*100:.1f}%)")
    else:
        print(f"📈 Revised supported: {total_v1_supported}/0 (0.0%)")
    print(f"🎯 Improvement: +{total_improvement} tasks")
    print(f"📁 Summary saved to: {summary_path}")
    
    if summary['failed_tasks'] > 0:
        print("\n❌ Failed analyses:")
        for result in summary['results']:
            if not result['result'].get('success'):
                print(f"  {result['model']}/{result['app']}: {result['result'].get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(main())
