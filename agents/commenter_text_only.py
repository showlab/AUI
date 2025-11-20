import json
from typing import List, Tuple
from pathlib import Path

from .base_commenter import BaseCommenter
from .prompts.commenter_prompts import build_text_only_prompt

class CommenterTextOnly(BaseCommenter):
    def _load_task_info(self, app_name: str, task_id: int, v0_dir: str = None) -> Tuple[str, str]:
        """从tasks.json加载任务信息"""
        if v0_dir:
            tasks_path = Path(f"initial/{v0_dir}/tasks/{app_name}/tasks.json")
        else:
            tasks_path = Path(f"tasks/{app_name}/tasks.json")
            
        with open(tasks_path, 'r', encoding='utf-8') as f:
            tasks_data = json.load(f)
        
        for task in tasks_data['tasks']:
            if task['id'] == task_id:
                return task['description'], task['expected_outcome']
        
        return f"Task {task_id}", "Unknown expected outcome"
    
    def _load_trajectory_text(self, trajectory_dir: str) -> List[dict]:
        """从trajectory.json提取文字信息"""
        trajectory_path = Path(trajectory_dir) / "trajectory.json"
        
        if not trajectory_path.exists():
            return []
        
        with open(trajectory_path, 'r', encoding='utf-8') as f:
            trajectory_data = json.load(f)
        
        steps = []
        for step_data in trajectory_data:
            step_num = step_data.get('step', 0)
            action_data = step_data.get('action', {})
            thought = step_data.get('thought', '')
            
            # 格式化action信息
            action_type = action_data.get('action', 'unknown')
            if action_type == 'left_click':
                coord = action_data.get('coordinate', [0, 0])
                action_text = f"Click({coord[0]},{coord[1]})"
            elif action_type == 'type':
                text = action_data.get('text', '')
                action_text = f"Type('{text}')"
            elif action_type == 'key':
                key = action_data.get('key', '')
                action_text = f"Key('{key}')"
            elif action_type == 'scroll':
                direction = action_data.get('direction', 'down')
                amount = action_data.get('amount', 3)
                action_text = f"Scroll({direction}, {amount})"
            else:
                action_text = str(action_data)
            
            steps.append({
                'step': step_num,
                'action': action_text,
                'thought': thought
            })
        
        return steps
    
    def _prepare_analysis_inputs(self, storyboard_path: str, html_content: str, website_screenshot: str, width: int, height: int) -> Tuple[str, List[str]]:
        """准备分析输入 - 使用纯文字信息"""
        # 从storyboard路径推导trajectory目录
        storyboard_path_obj = Path(storyboard_path)
        trajectory_dir = storyboard_path_obj.parent
        
        # 从路径提取app_name和task_id信息
        path_parts = trajectory_dir.parts
        app_name = None
        task_id = None
        v0_dir = None
        
        for i, part in enumerate(path_parts):
            if part == "tasks" and i + 1 < len(path_parts):
                app_name = path_parts[i + 1]
            elif part.startswith("task_"):
                task_id = int(part.split("_")[1])
            elif part == "initial" and i + 1 < len(path_parts):
                v0_dir = path_parts[i + 1]
        
        if not app_name or task_id is None:
            raise ValueError("Cannot extract task information from storyboard path")
        
        # 加载任务信息
        task_description, expected_outcome = self._load_task_info(app_name, task_id, v0_dir)
        
        # 加载轨迹文字信息
        trajectory_steps = self._load_trajectory_text(str(trajectory_dir))
        
        if not trajectory_steps:
            raise ValueError("No trajectory data found")
        
        # 构建文字描述
        steps_text = ""
        for step in trajectory_steps:
            steps_text += f"{step['step']}. Action: {step['action']}, Thought: {step['thought']}\n"
        
        # 构建分析prompt（要求结构化JSON输出）
        step_count = len(trajectory_steps)
        prompt = build_text_only_prompt(width, height, task_description, expected_outcome, steps_text, step_count)

        return prompt, [website_screenshot]
