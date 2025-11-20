#!/usr/bin/env python3
"""
Stage 0: 生成任务
使用GPT-5为每个应用生成30个任务，基于标签应用不同的测试哲学
"""

import argparse
import asyncio
import json
import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.model_client import ModelClient
from utils.parallel_runner import ParallelRunner
from utils.constants import DEFAULT_APPS
from agents.prompts.tasks_prompts import build_base_prompt, get_tag_based_prompt_template

# JSON example builder (ensures valid JSON in prompt)
def build_json_format_example(app_name: str, app_tags: list) -> str:
    example = {
        "app_name": app_name,
        "tags": app_tags,
        "tasks": [
            {
                "id": 1,
                "description": "Clear, specific task description",
                "category": "core_function|user_workflow|edge_case",
                "expected_outcome": "What should happen when task completes"
            }
        ]
    }
    return "Please respond in JSON format:\n" + json.dumps(example, ensure_ascii=False, indent=2)




async def generate_tasks_for_app(model_name: str, app_name: str, progress_tracker, initial_dir: str = "tasks", **kwargs) -> dict:
    """为单个应用生成任务（并行任务函数）"""
    model_client = ModelClient()
    
    progress_tracker.update_status(model_name, app_name, "📋 Loading instruction...")
    # 加载应用指令
    instruction_path = Path(f"examples/{app_name}.yaml")
    
    if not instruction_path.exists():
        raise FileNotFoundError(f"Instruction file not found: {instruction_path}")
    
    with open(instruction_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 严格依赖配置字段（不做默认回退）
    app_description = config['prompt']
    app_title = config['title']
    app_tags = config['tags']
    
    progress_tracker.update_status(model_name, app_name, f"✏️ Generating 30 tasks (tag: {app_tags[0]})...")
    
    # 获取基于标签的特定内容
    tag_specific_content = get_tag_based_prompt_template(app_tags)
    primary_tag = app_tags[0].lower()
    
    # 构建基础prompt
    base_prompt = build_base_prompt(
        tag_type=primary_tag,
        app_title=app_title,
        app_description=app_description,
        tag_specific_content=tag_specific_content,
        primary_tag=primary_tag,
    )
    
    # 构建JSON格式部分（严格为valid JSON示例）
    json_format = build_json_format_example(app_name, app_tags)
    
    # 组合完整prompt
    prompt = f"{base_prompt}\n\n{json_format}"
    
    # JSON解析重试机制
    tasks_data = None
    last_error = None
    
    for attempt in range(5):
        try:
            if attempt > 0:
                prompt += f"\n\nIMPORTANT: You must output valid JSON only. This is attempt {attempt + 1}/5."
            
            response = await model_client.call_task_generator(prompt)
            
            # 尝试解析JSON响应
            tasks_data = json.loads(response)
            break  # 成功解析，退出循环
            
        except json.JSONDecodeError as e:
            last_error = f"JSON parsing failed (attempt {attempt + 1}/5): {str(e)}\nResponse: {response[:200]}..."
            if attempt < 4:  # 不是最后一次尝试
                await asyncio.sleep(1)  # 稍等一下再重试
            continue
    
    if tasks_data is None:
        raise ValueError(f"Failed to parse JSON after 5 attempts. Last error: {last_error}")
    
    # 验证任务数量
    tasks = tasks_data.get('tasks', [])
    
    if not tasks:
        raise ValueError("No tasks generated - empty task list")
    
    progress_tracker.update_status(model_name, app_name, "💾 Saving tasks...")
    
    # 保存任务到文件
    tasks_file = save_tasks(app_name, tasks, app_tags, base_dir=initial_dir)
    
    return {
        'success': True,
        'tasks': tasks,
        'app_name': app_name,
        'tags': app_tags,
        'count': len(tasks),
        'tasks_file': tasks_file
    }

def save_tasks(app_name: str, tasks: list, tags: list, base_dir: str = "tasks") -> str:
    """保存任务到文件"""
    tasks_dir = Path(f"{base_dir}/{app_name}")
    tasks_dir.mkdir(parents=True, exist_ok=True)
    
    tasks_file = tasks_dir / "tasks.json"
    
    task_data = {
        'app_name': app_name,
        'tags': tags,
        'generated_at': str(Path(__file__).name),
        'task_count': len(tasks),
        'tasks': tasks
    }
    
    with open(tasks_file, 'w', encoding='utf-8') as f:
        json.dump(task_data, f, indent=2, ensure_ascii=False)
    
    return str(tasks_file)

async def main():
    parser = argparse.ArgumentParser(description='Generate tasks for apps')
    parser.add_argument('--apps', type=str, required=True,
                       help='Comma-separated list of apps or "all" for all 52 apps')
    parser.add_argument('--initial-dir', type=str, required=True,
                       help='Initial data directory name (stored under initial/)')
    
    args = parser.parse_args()
    
    # 解析应用列表
    if args.apps.lower() == 'all':
        apps = DEFAULT_APPS
    else:
        apps = args.apps.split(',')
    
    print(f"Stage 0: Generating tasks")
    print(f"Apps: {apps}")
    print(f"Using GPT-5 to generate 30 tasks per app with tag-based philosophy")
    print(f"Running {len(apps)} apps in parallel\n")
    
    # 创建initial目录结构  
    v0_base_path = f"initial/{args.initial_dir}/tasks"
    
    # 使用ParallelRunner并行生成任务
    runner = ParallelRunner(max_concurrent=5)  # GPT-5 only, no model parallelization needed
    
    # 生成任务（只用GPT-5）
    summary = await runner.run_parallel_tasks(
        models=['gpt5'],  # 只用GPT-5生成任务
        apps=apps,
        task_func=generate_tasks_for_app,
        stage_name="Stage 0: Generate Tasks",
        initial_dir=v0_base_path
    )
    
    # 统计结果
    successful_results = [r for r in summary['results'] if not r['result'].get('error')]
    total_tasks = sum(r['result'].get('count', 0) for r in successful_results)
    failed_results = [r for r in summary['results'] if r['result'].get('error')]
    
    # 保存总结（包括详细错误信息）
    summary_data = {
        'stage': 'Stage 0: Generate Tasks',
        'total_apps': len(apps),
        'successful_apps': len(successful_results),
        'failed_apps': len(failed_results),
        'total_tasks': total_tasks,
        'results': [{
            'app_name': r['app'],
            'success': not r['result'].get('error'),
            'task_count': r['result'].get('count', 0),
            'tasks_file': r['result'].get('tasks_file', ''),
            'error': r['result'].get('error', None),
            'full_error': r['result'].get('full_error', None)
        } for r in summary['results']],
        'errors': summary.get('errors', [])
    }
    
    base = Path(__file__).resolve().parent
    summary_path = base / "progress" / "global_summaries" / "summaries" / "stage0_tasks_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Summary:")
    print(f"✅ Successful: {summary_data['successful_apps']}/{len(apps)} apps")
    if summary_data['failed_apps'] > 0:
        print(f"❌ Failed: {summary_data['failed_apps']} apps")
    print(f"📝 Total tasks: {total_tasks}")
    print(f"📁 Summary saved to: {summary_path}")
    
    # 显示错误概览（如果有）
    if failed_results:
        print(f"\n❌ Failed Apps:")
        for result in failed_results:
            print(f"   - {result['app']}: {result['result'].get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(main())
