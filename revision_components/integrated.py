"""
Integrated Revision Component

Two-input merge revision approach:
Provides both unsupported tasks and CUA failure data simultaneously to the coder
for comprehensive revision in a single generation step.

This component combines both unsupported and CUA failure revision inputs together.
"""

import json
from typing import Dict, Any, List
from . import RevisionComponent
from .unsupported import UnsupportedTasksRevision
from .cua_failure import CuaFailureRevision

class IntegratedRevision(RevisionComponent):
    """Two-input merge revision: unsupported tasks + CUA failures processed simultaneously"""
    
    def __init__(self, coder, commenter=None, max_concurrent=10, revision_variant: str = 'integrated', commenter_variant: str = None):
        super().__init__(coder, commenter)
        self._revision_variant = revision_variant or 'integrated'
        self._commenter_variant = commenter_variant  # may be None; CuaFailureRevision will infer
        self.unsupported_component = UnsupportedTasksRevision(coder, commenter)
        # Ensure CUA failure component uses same commenter variant and its own revision variant context
        self.cua_failure_component = CuaFailureRevision(
            coder, commenter, max_concurrent=max_concurrent,
            revision_variant=self._revision_variant,
            commenter_variant=self._commenter_variant
        )
        # Bound integrated batch concurrency per model
        import asyncio
        self.max_concurrent = max_concurrent
        self._sema = asyncio.Semaphore(self.max_concurrent)
    
    async def revise(self, model_name: str, app_name: str, v0_html: str, 
                    context: Dict[str, Any], mcts: bool = False, 
                    destylized: bool = False, v0_dir: str = None, 
                    progress_tracker=None) -> Dict[str, Any]:
        """Two-input merge revision
        
        Args:
            model_name: Model for revision
            app_name: App name  
            v0_html: Original HTML
            context: Dict containing both 'failed_tasks' and optionally 'unsupported_tasks'
            mcts: Ignored (removed in new design)
            destylized: Apply destylization
            v0_dir: Initial data directory name (stored under initial/[dir])
        """
        import time
        
        # Load both input types
        failed_tasks = context.get('failed_tasks', [])
        unsupported_tasks = context.get('unsupported_tasks')
        
        # Load unsupported tasks if not provided in context
        if unsupported_tasks is None:
            unsupported_tasks = await self.unsupported_component._load_unsupported_tasks(model_name, app_name, v0_dir)
        
        # Check if we have any inputs to process
        if not failed_tasks and not unsupported_tasks:
            return {
                'success': True,
                'html_content': v0_html,
                'message': 'No failed tasks or unsupported tasks to fix',
                'revision_type': 'integrated_two_input',
                'failed_tasks_count': 0,
                'unsupported_tasks_count': 0
            }
        
        start_time = time.time()
        
        # Prepare failure analysis from CUA failures
        failure_analysis = ""
        if failed_tasks and self.commenter:
            # Generate CUA failure analyses using bounded concurrency in CUA component
            failure_analyses = await self.cua_failure_component._generate_all_comments(
                app_name=app_name,
                model_name=model_name,
                failed_tasks=failed_tasks,
                v0_html=v0_html,
                v0_dir=v0_dir,
                progress_tracker=progress_tracker
            )
            failure_analysis = self.cua_failure_component._combine_failure_analyses(failure_analyses)
        
        # Load task descriptions for unsupported tasks context
        task_descriptions = {}
        if unsupported_tasks:
            task_descriptions = await self.unsupported_component._load_task_descriptions(app_name, v0_dir)
        
        # Generate integrated revision using coder with both inputs
        result = await self._generate_integrated_revision(
            model_name=model_name,
            app_name=app_name,
            v0_html=v0_html,
            failed_tasks=failed_tasks,
            failure_analysis=failure_analysis,
            unsupported_tasks=unsupported_tasks,
            task_descriptions=task_descriptions,
            destylized=destylized,
            contract_prompt=context.get('non_regression_contract_prompt'),
            progress_tracker=progress_tracker
        )
        
        generation_time = time.time() - start_time
        
        result['revision_type'] = 'integrated_two_input'
        result['failed_tasks_count'] = len(failed_tasks)
        result['unsupported_tasks_count'] = len(unsupported_tasks) if unsupported_tasks else 0
        result['destylized'] = destylized
        result['generation_time'] = generation_time
        
        return result
    
    async def _generate_integrated_revision(self, model_name: str, app_name: str, 
                                          v0_html: str, failed_tasks: List[Dict], 
                                          failure_analysis: str, unsupported_tasks: List[Dict],
                                          task_descriptions: Dict, destylized: bool,
                                          contract_prompt: str = None,
                                          progress_tracker=None) -> Dict[str, Any]:
        """Generate integrated revision with both CUA failures and unsupported tasks"""
        
        # Use the existing coder method to generate the revised website and enhance the prompt
        # by combining failure_analysis with unsupported task information
        
        # Prepare unsupported tasks description
        unsupported_description = ""
        if unsupported_tasks:
            unsupported_list = []
            for task in unsupported_tasks:
                task_id = task.get('task_id', 0)
                task_desc = task_descriptions.get(task_id, {})
                description = task_desc.get('description', task.get('description', 'Unknown task'))
                reason = task.get('reason', 'No reason provided')
                if getattr(self, '_ablate_unsupported_reason', False):
                    unsupported_list.append(f"Task #{task_id}: {description}")
                else:
                    unsupported_list.append(f"Task #{task_id}: {description} - REASON: {reason}")
            
            unsupported_description = f"""
## UNSUPPORTED TASKS ANALYSIS
Tasks that the current website cannot support due to missing functionality:

{chr(10).join(unsupported_list)}

These tasks require additional UI elements, JavaScript functionality, or structural changes to be supported.
"""
        
        # Combine both analyses into comprehensive failure analysis
        combined_analysis = ""
        if failure_analysis and unsupported_description:
            combined_analysis = f"{failure_analysis}\n\n{unsupported_description}"
        elif failure_analysis:
            combined_analysis = failure_analysis
        elif unsupported_description:
            combined_analysis = unsupported_description
        else:
            combined_analysis = "No specific failure analysis available"
        
        # Generate revision using existing coder method with combined analysis
        result = await self.coder.generate_revised_website(
            model_name=model_name,
            app_name=app_name,
            v0_html=v0_html,
            failed_tasks=failed_tasks,
            failure_analysis=combined_analysis,
            apply_destylization=destylized,
            v0_dir=None,  # Not used in integrated mode
            non_regression_contract_prompt=contract_prompt,
            progress_tracker=progress_tracker
        )
        
        return result

    async def revise_model_batch(self, model_name: str, app_batch_data: List[Dict[str, Any]], 
                                 destylized: bool = False, v0_dir: str = None, progress_tracker=None,
                                 force_comments: bool = False,
                                 force_v1: bool = False,
                                 non_regression_prompts_by_app: Dict[str, str] = None) -> Dict[str, Any]:
        """Process multiple apps for a model concurrently for integrated revision"""
        import time
        import asyncio

        if not app_batch_data:
            return {'success': True, 'message': 'No apps to process', 'app_results': []}

        start_time = time.time()

        if progress_tracker:
            progress_tracker.update_status(model_name, "BATCH", f"📝 Integrated: {len(app_batch_data)} apps queued")

        async def run_single_app(app_data):
            app_name = app_data['app_name']
            v0_html = app_data['v0_html']
            failed_tasks = app_data.get('failed_tasks', []) or []

            async with self._sema:
                if progress_tracker:
                    progress_tracker.add_timing_info(model_name, "BATCH", f"{app_name}: ✏️ Generating revised (integrated)")

                # Prepare inputs: unsupported + optional CUA failure analyses
                unsupported_tasks = await self.unsupported_component._load_unsupported_tasks(model_name, app_name, v0_dir)
                task_descriptions = await self.unsupported_component._load_task_descriptions(app_name, v0_dir) if unsupported_tasks else {}

                failure_analysis = ""
                if failed_tasks and self.commenter:
                    analyses = await self.cua_failure_component._generate_all_comments(
                        app_name=app_name,
                        model_name=model_name,
                        failed_tasks=failed_tasks,
                        v0_html=v0_html,
                        v0_dir=v0_dir,
                        progress_tracker=progress_tracker,
                        force=force_comments
                    )
                    failure_analysis = self.cua_failure_component._combine_failure_analyses(analyses)

                contract_prompt = (non_regression_prompts_by_app or {}).get(app_name, "")

                result = await self._generate_integrated_revision(
                    model_name=model_name,
                    app_name=app_name,
                    v0_html=v0_html,
                    failed_tasks=failed_tasks,
                    failure_analysis=failure_analysis,
                    unsupported_tasks=unsupported_tasks,
                    task_descriptions=task_descriptions,
                    destylized=destylized,
                    contract_prompt=contract_prompt,
                    progress_tracker=progress_tracker
                )

                return {'app_name': app_name, 'result': result}

        tasks = [asyncio.create_task(run_single_app(app_data)) for app_data in app_batch_data]
        app_results = await asyncio.gather(*tasks)

        batch_time = time.time() - start_time
        successes = sum(1 for r in app_results if r.get('result', {}).get('success'))

        if progress_tracker:
            progress_tracker.update_status(model_name, "BATCH", f"✅ Integrated done: {successes}/{len(app_results)} in {batch_time:.1f}s")

        return {
            'success': successes > 0,
            'model': model_name,
            'processed_apps': len(app_batch_data),
            'batch_time': batch_time,
            'app_results': app_results
        }
