#!/usr/bin/env python3
"""
Stage 0: 生成初始网站
并行生成多个模型×多个 app 的初始网站
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.model_client import ModelClient
from utils.parallel_runner import ParallelRunner
from agents.coder import Coder
from utils.constants import DEFAULT_APPS

async def generate_website_task(model_name: str, app_name: str, progress_tracker, initial_dir: str = "websites", **kwargs) -> dict:
    """单个网站生成任务"""
    model_client = ModelClient()
    coder = Coder(model_client)
    
    progress_tracker.update_status(model_name, app_name, "📋 Loading instruction...")
    
    # 加载应用指令
    instruction = coder.load_app_instruction(app_name)
    
    progress_tracker.update_status(model_name, app_name, "✏️ Generating website...")
    
    # 生成网站（对GPT-5系列启用streaming，进度输出由Coder负责）
    html_content = await coder.generate_initial_website(
        model_name, app_name, instruction,
        progress_tracker=progress_tracker
    )
    
    progress_tracker.update_status(model_name, app_name, "💾 Saving website...")
    
    # 保存网站
    website_path = coder.save_website(html_content, app_name, model_name, phase="initial", base_dir=initial_dir)
    
    return {
        'website_path': website_path,
        'model': model_name,
        'app': app_name,
        'success': True
    }

async def main():
    parser = argparse.ArgumentParser(description='Generate initial websites')
    parser.add_argument('--models', type=str, required=True,
                       help='Comma-separated list of models (e.g., gpt5,qwen,gpt4o)')
    parser.add_argument('--apps', type=str, required=True,
                       help='Comma-separated list of apps or "all" for all 52 apps')
    parser.add_argument('--max-concurrent', type=int, default=5,
                       help='Maximum concurrent tasks')
    parser.add_argument('--initial-dir', type=str, required=True,
                       help='Initial data directory name (stored under initial/)')
    
    args = parser.parse_args()
    
    # 解析模型列表
    models = args.models.split(',')
    
    # 解析应用列表
    if args.apps.lower() == 'all':
        apps = DEFAULT_APPS
    else:
        apps = args.apps.split(',')
    
    print(f"Stage 0: Generating initial websites")
    print(f"Models: {models}")
    print(f"Apps: {apps}")
    print(f"Total tasks: {len(models)} × {len(apps)} = {len(models) * len(apps)}")
    
    # 创建并行执行器
    runner = ParallelRunner(max_concurrent=args.max_concurrent)
    
    # 创建v0目录结构
    v0_base_path = f"initial/{args.initial_dir}/websites"
    
    # 运行任务
    summary = await runner.run_parallel_tasks(
        models=models,
        apps=apps,
        task_func=generate_website_task,
        stage_name="Stage 0: Generate Initial Websites",
        initial_dir=v0_base_path
    )
    
    # 保存总结（包括详细错误信息）
    import json
    
    successful_count = summary['successful_tasks']
    failed_count = summary['failed_tasks']
    
    # 增强的summary数据
    enhanced_summary = {
        'stage': summary['stage'],
        'total_tasks': summary['total_tasks'],
        'successful_tasks': successful_count,
        'failed_tasks': failed_count,
        'models': models,
        'apps': apps,
        'results': summary['results'],
        'errors': summary.get('errors', [])
    }
    
    base = Path(__file__).resolve().parent
    summary_path = base / "progress" / "global_summaries" / "summaries" / "stage0_websites_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Summary:")
    print(f"✅ Successful: {successful_count}/{summary['total_tasks']} tasks")
    if failed_count > 0:
        print(f"❌ Failed: {failed_count} tasks")
    print(f"📁 Summary saved to: {summary_path}")

if __name__ == "__main__":
    asyncio.run(main())
