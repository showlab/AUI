#!/usr/bin/env python3
"""
Stage 2: CUA 测试初始网站
使用 UI‑TARS 7B 在初始网站上执行任务
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
from agents.cua_policy import create_cua_policy
from utils.constants import DEFAULT_APPS

async def cua_test_task(model_name: str, app_name: str, progress_tracker, initial_dir: str = None, cua_model: str = "uitars", **kwargs) -> dict:
    """单个CUA测试任务"""
    try:
        model_client = ModelClient()
        cua_policy = create_cua_policy(model_client, cua_model_name=cua_model, max_steps=20)
        
        progress_tracker.update_status(model_name, app_name, "Loading rules...")
        
        # 加载规则
        if initial_dir:
            rules_path = Path(f"initial/{initial_dir}/tasks/{app_name}/states/{model_name}/rules.json")
            website_path = Path(f"initial/{initial_dir}/websites/{app_name}/{model_name}/index.html")
        else:
            rules_path = Path(f"tasks/{app_name}/states/{model_name}/rules.json")
            website_path = Path(f"websites/{app_name}/{model_name}/index.html")
        if not rules_path.exists():
            return {
                'success': False,
                'error': f"Rules not found: {rules_path}",
                'model': model_name,
                'app': app_name
            }
        
        with open(rules_path, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
        
        # 获取支持的任务
        supported_tasks = rules_data.get('analysis', {}).get('supported_tasks', [])
        
        if not supported_tasks:
            return {
                'success': True,
                'completed_tasks': 0,
                'total_tasks': 0,
                'results': [],
                'message': "No supported tasks found",
                'model': model_name,
                'app': app_name
            }
        
        progress_tracker.update_status(model_name, app_name, f"Testing {len(supported_tasks)} tasks...")
        
        # 网站URL
        website_url = f"file://{website_path.absolute()}"
        
        # 执行任务
        task_results = []
        completed_count = 0
        
        # 加载任务描述映射
        if initial_dir:
            tasks_file = f"initial/{initial_dir}/tasks/{app_name}/tasks.json"
            trajectories_dir = Path(f"initial/{initial_dir}/tasks/{app_name}/initial_cua_results/{model_name}/{cua_model}/trajectories")
        else:
            tasks_file = f"tasks/{app_name}/tasks.json"
            trajectories_dir = Path(f"tasks/{app_name}/initial_cua_results/{model_name}/{cua_model}/trajectories")
            
        with open(tasks_file) as f:
            tasks_data = json.load(f)
        task_map = {t['id']: t['description'] for t in tasks_data['tasks']}
        
        # 创建保存轨迹的目录
        trajectories_dir.mkdir(parents=True, exist_ok=True)
        
        for i, task_info in enumerate(supported_tasks):
            task_id = task_info['task_index']
            task_description = task_map[task_id]
            completion_rule = task_info.get('rule', '')
            
            progress_tracker.update_status(
                model_name, app_name, 
                f"Task {i+1}/{len(supported_tasks)}: {task_description[:30]}..."
            )
            
            # 为每个任务创建单独的轨迹目录
            task_trajectory_dir = trajectories_dir / f"task_{i+1}"
            
            # 清理旧轨迹文件以防止污染
            if task_trajectory_dir.exists():
                for old_file in task_trajectory_dir.glob("step_*.png"):
                    old_file.unlink()
                for old_file in task_trajectory_dir.glob("*.json"):
                    old_file.unlink()
            
            task_trajectory_dir.mkdir(parents=True, exist_ok=True)
            
            # 执行任务
            result = await cua_policy.execute_task(
                app_name=app_name,
                model_name=model_name,
                website_url=website_url,
                task={'description': task_description},
                completion_rule=completion_rule,
                save_dir=str(task_trajectory_dir)
            )
            
            # Save trajectory data to file for storyboard generation
            trajectory_file = task_trajectory_dir / "trajectory.json"
            if 'trajectory' in result:
                with open(trajectory_file, 'w', encoding='utf-8') as f:
                    json.dump(result['trajectory'], f, indent=2, ensure_ascii=False)
            
            task_results.append({
                'task_index': i + 1,
                'task_description': task_description,
                'success': result['success'],
                'completed': result.get('completed', False),
                'steps': result.get('steps', 0),
                'trajectory_dir': str(task_trajectory_dir),
                'error': result.get('error', None)
            })
            
            if result.get('completed', False):
                completed_count += 1
        
        # 保存结果
        results_data = {
            'app_name': app_name,
            'model_name': model_name,
            'total_supported_tasks': len(supported_tasks),
            'tested_tasks': len(task_results),
            'completed_tasks': completed_count,
            'success_rate': completed_count / len(task_results) if task_results else 0,
            'task_results': task_results
        }
        
        if initial_dir:
            results_dir = Path(f"initial/{initial_dir}/tasks/{app_name}/initial_cua_results/{model_name}/{cua_model}")
        else:
            results_dir = Path(f"tasks/{app_name}/initial_cua_results/{model_name}/{cua_model}")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / "results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        return {
            'success': True,
            'completed_tasks': completed_count,
            'total_tasks': len(task_results),
            'success_rate': completed_count / len(task_results) if task_results else 0,
            'results_file': str(results_file),
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
    parser = argparse.ArgumentParser(description='CUA test initial websites')
    parser.add_argument('--models', type=str, required=True,
                       help='Comma-separated list of models (e.g., gpt5,qwen,gpt4o)')
    parser.add_argument('--apps', type=str, required=True,
                       help='Comma-separated list of apps or "all" for all 52 apps')
    parser.add_argument('--max-concurrent', type=int, default=20,
                       help='Maximum concurrent tasks (limited due to browser resources)')
    parser.add_argument('--initial-dir', type=str, default=None,
                       help='Initial data directory name (stored under initial/[dir])')
    parser.add_argument('--cua-models', type=str, default='uitars',
                       help='Comma-separated list of CUA models (e.g., uitars,operator)')
    
    args = parser.parse_args()
    
    # 解析模型列表
    models = args.models.split(',')
    cua_models = args.cua_models.split(',')
    
    # 解析应用列表
    if args.apps.lower() == 'all':
        apps = DEFAULT_APPS
    else:
        apps = args.apps.split(',')
    
    print(f"Stage 2: CUA testing initial websites")
    print(f"Models: {models}")
    print(f"Apps: {apps}")
    print(f"CUA models: {cua_models}")
    
    # 验证必要文件存在
    missing_files = []
    for model in models:
        for app in apps:
            if args.initial_dir:
                website_path = Path(f"initial/{args.initial_dir}/websites/{app}/{model}/index.html")
                rules_path = Path(f"initial/{args.initial_dir}/tasks/{app}/states/{model}/rules.json")
            else:
                website_path = Path(f"websites/{app}/{model}/index.html")
                rules_path = Path(f"tasks/{app}/states/{model}/rules.json")
            
            if not website_path.exists():
                missing_files.append(f"Website: {website_path}")
            if not rules_path.exists():
                missing_files.append(f"Rules: {rules_path}")
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  {file}")
        return
    
    # 创建并行执行器
    runner = ParallelRunner(max_concurrent=args.max_concurrent)
    
    # 为每个CUA模型运行任务
    all_results = []
    for cua_model in cua_models:
        print(f"\n🤖 Testing with CUA model: {cua_model}")
        summary = await runner.run_parallel_tasks(
            models=models,
            apps=apps,
            task_func=cua_test_task,
            initial_dir=args.initial_dir,
            cua_model=cua_model,
            stage_name=f"Stage 2: CUA Test Initial Websites ({cua_model})"
        )
        all_results.append({
            'cua_model': cua_model,
            'summary': summary
        })
    
    # 计算统计信息 (跨所有CUA模型)
    total_completed = 0
    total_tested = 0
    
    for cua_result in all_results:
        for result in cua_result['summary']['results']:
            if result['result'].get('success'):
                total_completed += result['result'].get('completed_tasks', 0)
                total_tested += result['result'].get('total_tasks', 0)
    
    # 保存总结 (包含所有CUA模型结果)
    detailed_summary = {
        'cua_models': cua_models,
        'all_results': all_results,
        'total_completed_tasks': total_completed,
        'total_tested_tasks': total_tested,
        'overall_success_rate': total_completed / total_tested if total_tested > 0 else 0
    }
    
    base = Path(__file__).resolve().parent
    summary_path = base / "progress" / "global_summaries" / "summaries" / "stage2_cua_test_initial_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_summary, f, indent=2, ensure_ascii=False)
    
    # Generate detailed evaluation output
    eval_dir = base / "progress" / "global_summaries" / "evaluations" / "stage2_cua_test_initial_eval"
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
                completed = result['result'].get('completed_tasks', 0)
                total = result['result'].get('total_tasks', 0)
                success_rate = result['result'].get('success_rate', 0)
                
                # Create combined key for model+cua_model
                model_key = f"{model}+{cua_model}"
                
                if model_key not in model_stats:
                    model_stats[model_key] = {'completed': 0, 'total': 0, 'apps': []}
                model_stats[model_key]['completed'] += completed
                model_stats[model_key]['total'] += total
                model_stats[model_key]['apps'].append({
                    'app': app,
                    'completed': completed,
                    'total': total,
                    'success_rate': success_rate
                })
            
            if app not in app_stats:
                app_stats[app] = {'models': []}
            app_stats[app]['models'].append({
                'model': model,
                'completed': completed,
                'total': total,
                'success_rate': success_rate
            })
    
    # Save model comparison
    model_comparison = {
        'overview': {
            model: {
                'total_completed': stats['completed'],
                'total_tested': stats['total'],
                'overall_success_rate': stats['completed'] / stats['total'] if stats['total'] > 0 else 0,
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
            'summary': {
                'total_completed': stats['completed'],
                'total_tested': stats['total'],
                'overall_success_rate': stats['completed'] / stats['total'] if stats['total'] > 0 else 0,
                'num_apps_tested': len(stats['apps'])
            },
            'apps': stats['apps']
        }
        
        with open(eval_dir / f"{model}_detailed.json", 'w', encoding='utf-8') as f:
            json.dump(model_detailed, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 CUA Test Summary (All CUA Models):")
    total_successful = sum(cua_result['summary']['successful_tasks'] for cua_result in all_results)
    total_tasks = sum(cua_result['summary']['total_tasks'] for cua_result in all_results)
    print(f"✅ Successful tests: {total_successful}/{total_tasks}")
    if total_tested > 0:
        print(f"🎯 Completed tasks: {total_completed}/{total_tested} ({total_completed/total_tested*100:.1f}%)")
    else:
        print(f"🎯 Completed tasks: {total_completed}/0 (0.0%)")
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
