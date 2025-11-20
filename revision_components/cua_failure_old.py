"""
CUA Failure Revision Component

Revises websites based on CUA policy failure trajectories.
Supports MCTS (10 versions + commenter selection) and destylization.
"""

import json
import time
from typing import Dict, Any, List
from pathlib import Path
from . import RevisionComponent

class CuaFailureRevision(RevisionComponent):
    """Revision based on CUA policy failures"""
    
    async def revise(self, model_name: str, app_name: str, v0_html: str, 
                    context: Dict[str, Any], mcts: bool = False, 
                    destylized: bool = False, v0_dir: str = None) -> Dict[str, Any]:
        """Revise based on CUA failure trajectories
        
        Args:
            model_name: Model for revision
            app_name: App name
            v0_html: Original HTML
            context: Dict with 'failed_tasks' key containing failure data
            mcts: Generate 10 versions and select best
            destylized: Apply destylization modifications
        """
        failed_tasks = context.get('failed_tasks', [])
        if not failed_tasks:
            return {
                'success': True,
                'html_content': v0_html,
                'message': 'No failed tasks to fix',
                'revision_type': 'cua_failure',
                'mcts_used': False,
                'destylized': False
            }
        
        if mcts:
            return await self._generate_mcts_versions(
                model_name, app_name, v0_html, failed_tasks, destylized, v0_dir
            )
        else:
            return await self._generate_single_version(
                model_name, app_name, v0_html, failed_tasks, destylized, v0_dir
            )
    
    async def _generate_single_version(self, model_name: str, app_name: str, 
                                     v0_html: str, failed_tasks: List[Dict], 
                                     destylized: bool, v0_dir: str = None) -> Dict[str, Any]:
        """Generate single revised version with failure analysis"""
        # Use commenter to analyze CUA failures first
        failure_analysis = await self.commenter.analyze_cua_failures(
            model_name=model_name,
            app_name=app_name,
            failed_tasks=failed_tasks,
            html_content=v0_html,
            v0_dir=v0_dir
        )
        
        result = await self.coder.generate_revised_website(
            model_name=model_name,
            app_name=app_name,
            v0_html=v0_html,
            failed_tasks=failed_tasks,
            failure_analysis=failure_analysis,
            apply_destylization=destylized,
            v0_dir=v0_dir
        )
        
        result['revision_type'] = 'cua_failure'
        result['mcts_used'] = False
        result['destylized'] = destylized
        result['failure_analysis'] = failure_analysis
        return result
    
    async def _generate_mcts_versions(self, model_name: str, app_name: str, 
                                    v0_html: str, failed_tasks: List[Dict], 
                                    destylized: bool, v0_dir: str = None) -> Dict[str, Any]:
        """Generate 10 versions using MCTS and select best with failure analysis"""
        # Use commenter to analyze CUA failures first
        failure_analysis = await self.commenter.analyze_cua_failures(
            model_name=model_name,
            app_name=app_name,
            failed_tasks=failed_tasks,
            html_content=v0_html,
            v0_dir=v0_dir
        )
        
        versions = []
        generation_details = []
        
        # Generate 3 versions with same prompt but different temperature/attempts
        for i in range(3):
            start_time = time.time()
            result = await self.coder.generate_revised_website(
                model_name=model_name,
                app_name=app_name,
                v0_html=v0_html,
                failed_tasks=failed_tasks,
                failure_analysis=failure_analysis,
                apply_destylization=destylized,
                v0_dir=v0_dir
            )
            generation_time = time.time() - start_time
            
            versions.append(result['html_content'])
            generation_details.append({
                'version': i,
                'success': result['success'],
                'generation_time': generation_time,
                'html_length': len(result['html_content']) if result['success'] else 0,
                'error': result.get('error') if not result['success'] else None
            })
        
        
        # Use commenter to select best version
        task_context = [{'description': task.get('description', '')} 
                       for task in failed_tasks]
        
        selected_idx = await self.commenter.select_best_version(
            model_name=model_name,
            app_name=app_name,
            html_versions=versions,
            task_context=task_context
        )
        
        return {
            'success': True,
            'html_content': versions[selected_idx],
            'selected_version': selected_idx,
            'revision_type': 'cua_failure',
            'mcts_used': True,
            'destylized': destylized,
            'generation_details': generation_details,
            'failed_tasks_analyzed': len(failed_tasks),
            'failure_analysis': failure_analysis
        }
