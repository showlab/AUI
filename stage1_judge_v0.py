#!/usr/bin/env python3
"""
Stage 1: Judge评估初始网站
使用Judge模型分析初始网站，提取状态规则
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
from agents.judge import Judge
from utils.constants import DEFAULT_APPS

async def judge_website_task(model_name: str, app_name: str, progress_tracker, initial_dir: str = None, **kwargs) -> dict:
    """单个网站judge任务"""
    try:
        model_client = ModelClient()
        judge = Judge(model_client)
        
        progress_tracker.update_status(model_name, app_name, "Loading website...")
        
        # 加载网站HTML
        if initial_dir:
            website_path = Path(f"initial/{initial_dir}/websites/{app_name}/{model_name}/index.html")
            tasks_path = Path(f"initial/{initial_dir}/tasks/{app_name}/tasks.json")
        else:
            website_path = Path(f"websites/{app_name}/{model_name}/index.html")
            tasks_path = Path(f"tasks/{app_name}/tasks.json")
            
        if not website_path.exists():
            return {
                'success': False,
                'error': f"Website not found: {website_path}",
                'model': model_name,
                'app': app_name
            }
        
        with open(website_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        progress_tracker.update_status(model_name, app_name, "Loading tasks...")
        
        # 加载任务
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
        
        progress_tracker.update_status(model_name, app_name, "Analyzing website...")
        
        # 分析网站和任务 - 单一规则输出（无组件拆分）
        analysis_result = await judge.analyze_website_tasks(app_name, html_content, tasks)
        
        if analysis_result['success']:
            progress_tracker.update_status(model_name, app_name, "Saving rules...")
            
            # 保存规则（单一 rules.json）
            rules_file = judge.save_rules(app_name, model_name, analysis_result, version="initial", v0_dir=initial_dir)
            
            return {
                'success': True,
                'analysis': analysis_result['analysis'],
                'supported_count': analysis_result['supported_count'],
                'unsupported_count': analysis_result['unsupported_count'],
                'total_tasks': analysis_result['total_tasks'],
                'rules_file': rules_file,
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
    parser = argparse.ArgumentParser(description='Judge evaluate initial websites')
    parser.add_argument('--models', type=str, required=True,
                       help='Comma-separated list of models (e.g., gpt5,qwen,gpt4o)')
    parser.add_argument('--apps', type=str, required=True,
                       help='Comma-separated list of apps or "all" for all 52 apps')
    parser.add_argument('--max-concurrent', type=int, default=10,
                       help='Maximum concurrent tasks (limited due to judge model API)')
    parser.add_argument('--initial-dir', type=str, default=None,
                       help='Initial data directory name (stored under initial/[dir])')
    
    args = parser.parse_args()
    
    # 解析模型列表
    models = args.models.split(',')
    
    # 解析应用列表
    if args.apps.lower() == 'all':
        apps = DEFAULT_APPS
    else:
        apps = args.apps.split(',')
    
    print(f"Stage 1: Judge evaluating initial websites")
    print(f"Models: {models}")
    print(f"Apps: {apps}")
    print(f"Using GPT-5 to analyze websites and extract rules")
    
    # 验证网站和任务文件存在
    missing_files = []
    for model in models:
        for app in apps:
            if args.initial_dir:
                website_path = Path(f"initial/{args.initial_dir}/websites/{app}/{model}/index.html")
                tasks_path = Path(f"initial/{args.initial_dir}/tasks/{app}/tasks.json")
            else:
                website_path = Path(f"websites/{app}/{model}/index.html")
                tasks_path = Path(f"tasks/{app}/tasks.json")
            
            if not website_path.exists():
                missing_files.append(f"Website: {website_path}")
            if not tasks_path.exists():
                missing_files.append(f"Tasks: {tasks_path}")
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  {file}")
        return
    
    # 创建并行执行器
    runner = ParallelRunner(max_concurrent=args.max_concurrent)
    
    # 运行任务
    summary = await runner.run_parallel_tasks(
        models=models,
        apps=apps,
        task_func=judge_website_task,
        stage_name="Stage 1: Judge Initial Websites",
        initial_dir=args.initial_dir
    )
    
    # 计算统计信息
    total_supported = 0
    total_tasks = 0
    
    for result in summary['results']:
        if result['result'].get('success'):
            total_supported += result['result'].get('supported_count', 0)
            total_tasks += result['result'].get('total_tasks', 0)
    
    # 保存总结
    detailed_summary = {
        **summary,
        'total_supported_tasks': total_supported,
        'total_tasks_analyzed': total_tasks,
        'support_rate': total_supported / total_tasks if total_tasks > 0 else 0
    }
    
    base = Path(__file__).resolve().parent
    summary_path = base / "progress" / "global_summaries" / "summaries" / "stage1_judge_initial_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_summary, f, indent=2, ensure_ascii=False)
    
    # Generate detailed evaluation output
    eval_dir = base / "progress" / "global_summaries" / "evaluations" / "stage1_judge_initial_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Model comparison analysis
    model_stats = {}
    app_stats = {}
    
    for result in summary['results']:
        if result['result'].get('success'):
            model = result['model']
            app = result['app']
            supported = result['result'].get('supported_count', 0)
            total = result['result'].get('total_tasks', 0)
            
            if model not in model_stats:
                model_stats[model] = {'supported': 0, 'total': 0, 'apps': []}
            model_stats[model]['supported'] += supported
            model_stats[model]['total'] += total
            model_stats[model]['apps'].append({
                'app': app,
                'supported': supported,
                'total': total,
                'rate': supported / total if total > 0 else 0
            })
            
            if app not in app_stats:
                app_stats[app] = {'models': []}
            app_stats[app]['models'].append({
                'model': model,
                'supported': supported,
                'total': total,
                'rate': supported / total if total > 0 else 0
            })
    
    # Save model comparison
    model_comparison = {
        'overview': {
            model: {
                'total_supported': stats['supported'],
                'total_tasks': stats['total'],
                'support_rate': stats['supported'] / stats['total'] if stats['total'] > 0 else 0,
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
                'total_supported': stats['supported'],
                'total_tasks': stats['total'],
                'support_rate': stats['supported'] / stats['total'] if stats['total'] > 0 else 0,
                'num_apps_tested': len(stats['apps'])
            },
            'apps': stats['apps']
        }
        
        with open(eval_dir / f"{model}_detailed.json", 'w', encoding='utf-8') as f:
            json.dump(model_detailed, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Analysis Summary:")
    print(f"✅ Successful analyses: {summary['successful_tasks']}/{summary['total_tasks']}")
    if total_tasks > 0:
        print(f"📝 Supported tasks: {total_supported}/{total_tasks} ({total_supported/total_tasks*100:.1f}%)")
    else:
        print(f"📝 Supported tasks: {total_supported}/0 (0.0%)")
    print(f"📁 Summary saved to: {summary_path}")
    
    if summary['failed_tasks'] > 0:
        print("\n❌ Failed analyses:")
        for result in summary['results']:
            if not result['result'].get('success'):
                print(f"  {result['model']}/{result['app']}: {result['result'].get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(main())
