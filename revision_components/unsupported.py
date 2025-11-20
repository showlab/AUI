"""
Unsupported Tasks Revision Component

Revises websites to support tasks that were marked as unsupported by Stage 1 judge.
Focuses on adding missing UI elements and functionality.
"""

import json
import time
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
from . import RevisionComponent

class UnsupportedTasksRevision(RevisionComponent):
    """Revision based on unsupported tasks from Stage 1 judge"""

    def __init__(self, coder, commenter=None, max_concurrent: int = 10):
        super().__init__(coder, commenter)
        # Bound concurrent generations per model (like integrated)
        self.max_concurrent = max_concurrent
        self._sema = asyncio.Semaphore(self.max_concurrent)
    
    async def revise(self, model_name: str, app_name: str, v0_html: str, 
                    context: Dict[str, Any], mcts: bool = False, 
                    destylized: bool = False, v0_dir: str = None, 
                    progress_tracker=None) -> Dict[str, Any]:
        """Revise to support unsupported tasks
        
        Args:
            model_name: Model for revision  
            app_name: App name
            v0_html: Original HTML
            context: Dict with 'unsupported_tasks' or loads from Stage 1 results
            mcts: Ignored for unsupported revision
            destylized: Ignored for unsupported revision
        """
        # Load unsupported tasks from context or judge results
        unsupported_tasks = context.get('unsupported_tasks')
        
        if unsupported_tasks is None:
            unsupported_tasks = await self._load_unsupported_tasks(model_name, app_name, v0_dir)
        
        if not unsupported_tasks:
            return {
                'success': True,
                'html_content': v0_html,
                'message': 'No unsupported tasks to fix',
                'revision_type': 'unsupported_tasks',
                'unsupported_tasks_count': 0
            }
        
        # Load task descriptions for context
        task_descriptions = await self._load_task_descriptions(app_name, v0_dir)
        
        # Generate unsupported task revision using coder
        start_time = time.time()
        ablate_reason = getattr(self, '_ablate_unsupported_reason', False)
        ablate_desc_only = getattr(self, '_ablate_desc_only', False)
        ablate_no_contract = getattr(self, '_ablate_no_contract', False)
        result = await self.coder.generate_unsupported_revision(
            model_name=model_name,
            app_name=app_name,
            v0_html=v0_html,
            unsupported_tasks=unsupported_tasks,
            task_descriptions=task_descriptions,
            non_regression_contract_prompt=context.get('non_regression_contract_prompt'),
            ablate_unsupported_reason=ablate_reason,
            ablate_desc_only=ablate_desc_only,
            ablate_no_contract=ablate_no_contract
        )
        generation_time = time.time() - start_time
        
        result['revision_type'] = 'unsupported_tasks'
        result['unsupported_tasks_count'] = len(unsupported_tasks)
        result['generation_time'] = generation_time
        
        return result

    async def revise_model_batch(self, model_name: str, app_batch_data: List[Dict[str, Any]], 
                                 destylized: bool = False, v0_dir: str = None, progress_tracker=None,
                                 non_regression_prompts_by_app: Optional[Dict[str, str]] = None,
                                 **kwargs) -> Dict[str, Any]:
        """Process multiple apps for a model concurrently for unsupported-task revision
        - Mirrors integrated/cua batch shape and BATCH progress updates
        - Returns per-app results with html_content for outer saver
        """
        import time as _time

        if not app_batch_data:
            return {'success': True, 'message': 'No apps to process', 'app_results': []}

        start_time = _time.time()

        if progress_tracker:
            progress_tracker.update_status(model_name, "BATCH", f"📝 Unsupported: {len(app_batch_data)} apps queued")

        async def run_single_app(app_data: Dict[str, Any]):
            app_name_l = app_data['app_name']
            v0_html_l = app_data['v0_html']
            # Always load and normalize unsupported tasks like integrated component does
            # Ignore raw items from app_batch_data to ensure consistent schema
            unsupported_tasks_l = await self._load_unsupported_tasks(model_name, app_name_l, v0_dir)

            async with self._sema:
                if progress_tracker:
                    progress_tracker.add_timing_info(model_name, "BATCH", f"{app_name_l}: ✏️ Generating revised (unsupported)")

                # If no unsupported tasks, return initial site as-is (consistent with single revise)
                if not unsupported_tasks_l:
                    res = {
                        'success': True,
                        'html_content': v0_html_l,
                        'revision_type': 'unsupported_tasks',
                        'unsupported_tasks_count': 0,
                        'generation_time': 0.0,
                        'destylized': destylized,
                        'analyzed_failures': 0
                    }
                    return {'app_name': app_name_l, 'result': res}

                # Load task descriptions for prompt context
                task_descriptions = await self._load_task_descriptions(app_name_l, v0_dir)
                contract_prompt = (non_regression_prompts_by_app or {}).get(app_name_l, "")

                # Generate revised with unsupported-task improvements
                result = await self.coder.generate_unsupported_revision(
                    model_name=model_name,
                    app_name=app_name_l,
                    v0_html=v0_html_l,
                    unsupported_tasks=unsupported_tasks_l,
                    task_descriptions=task_descriptions,
                    non_regression_contract_prompt=contract_prompt,
                    ablate_unsupported_reason=getattr(self, '_ablate_unsupported_reason', False),
                    ablate_desc_only=getattr(self, '_ablate_desc_only', False),
                    ablate_no_contract=getattr(self, '_ablate_no_contract', False)
                )
                # Mark counts and revision type for outer summary (align with integrated)
                result['revision_type'] = 'unsupported_tasks'
                result['unsupported_tasks_count'] = len(unsupported_tasks_l)
                result['destylized'] = destylized
                result['analyzed_failures'] = 0
                return {'app_name': app_name_l, 'result': result}

        # Launch with progress updates like CUA/Integrated
        total = len(app_batch_data)
        done = 0
        successes = 0
        app_results: List[Dict[str, Any]] = []
        tasks = [run_single_app(app_data) for app_data in app_batch_data]
        for fut in asyncio.as_completed(tasks):
            res = await fut
            app_results.append(res)
            done += 1
            if res.get('result', {}).get('success'):
                successes += 1
            if progress_tracker:
                progress_tracker.update_status(model_name, "BATCH", f"✏️ Unsupported: {done}/{total} websites")

        batch_time = _time.time() - start_time

        if progress_tracker:
            progress_tracker.update_status(model_name, "BATCH", f"✅ Unsupported done: {successes}/{total} in {batch_time:.1f}s")

        return {
            'success': successes > 0,
            'model': model_name,
            'processed_apps': total,
            'batch_time': batch_time,
            'app_results': app_results
        }
    
    async def _load_unsupported_tasks(self, model_name: str, app_name: str, v0_dir: str = None) -> List[Dict[str, Any]]:
        """Load unsupported tasks from Stage 1 judge results"""
        if v0_dir:
            rules_path = Path(f"initial/{v0_dir}/tasks/{app_name}/states/{model_name}/rules.json")
        else:
            rules_path = Path(f"tasks/{app_name}/states/{model_name}/rules.json")
        
        if not rules_path.exists():
            return []
        
        with open(rules_path, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
        
        unsupported_tasks = []
        for task_data in rules_data.get('analysis', {}).get('unsupported_tasks', []):
            if not task_data.get('supportable', True):
                unsupported_tasks.append({
                    'task_id': task_data['task_index'],
                    'reason': task_data.get('reason', 'Unknown reason'),
                    'supported': False
                })
        
        return unsupported_tasks
    
    async def _load_task_descriptions(self, app_name: str, v0_dir: str = None) -> Dict[int, Dict[str, Any]]:
        """Load task descriptions for context"""
        if v0_dir:
            tasks_file = Path(f"initial/{v0_dir}/tasks/{app_name}/tasks.json")
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
