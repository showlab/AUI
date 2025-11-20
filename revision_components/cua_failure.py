"""
CUA Failure Revision Component (thin wrapper)

This wrapper preserves the public API but delegates heavy logic to
revision_components.cua_failure_helpers to keep the file size small
without changing behavior.
"""

from typing import Dict, Any, List
from pathlib import Path
from . import RevisionComponent

# Delegate helpers
from .cua_failure_batch import process_model_batch, generate_revised_version


class CuaFailureRevision(RevisionComponent):
    """Revision based on CUA policy failures using storyboard analysis"""

    def __init__(self, coder, commenter=None, max_concurrent=10,
                 revision_variant: str = 'cua', commenter_variant: str = None):
        super().__init__(coder, commenter)
        self.max_concurrent = max_concurrent

        # Concurrency for commenter calls
        import asyncio
        self._commenter_sema = asyncio.Semaphore(self.max_concurrent)

        # Anchor IO to project root
        self._base_dir = Path(__file__).resolve().parents[1]

        # Cache helpers and naming variants
        from utils.cache_paths import (
            comment_cache_dir as _ccd,
            revised_cache_dir as _revised_d,
            commenter_variant_from_instance as _cvfi,
        )
        self._comment_cache_dir = _ccd
        self._revised_cache_dir = _revised_d
        self._revision_variant = revision_variant or 'cua'
        self._commenter_variant = (commenter_variant or _cvfi(commenter))

    async def revise(self, model_name: str, app_name: str, v0_html: str,
                     context: Dict[str, Any], destylized: bool = False,
                     v0_dir: str = None, progress_tracker=None) -> Dict[str, Any]:
        failed_tasks = context.get('failed_tasks', [])
        if not failed_tasks:
            # Generate destylized revised website even without failures
            result = await self.coder.generate_revised_website(
                model_name=model_name,
                app_name=app_name,
                v0_html=v0_html,
                failed_tasks=[],
                failure_analysis="No CUA failures available; apply destylization and fit within screen.",
                apply_destylization=destylized,
                v0_dir=v0_dir,
                progress_tracker=progress_tracker,
                non_regression_contract_prompt=context.get('non_regression_contract_prompt')
            )
            result['analyzed_failures'] = 0
            result['revision_type'] = 'cua_failure'
            result['destylized'] = destylized
            return result

        return await generate_revised_version(
            self, model_name, app_name, v0_html, failed_tasks,
            destylized, v0_dir, progress_tracker,
            contract_prompt=context.get('non_regression_contract_prompt')
        )

    async def revise_model_batch(self, model_name: str, app_batch_data: List[Dict[str, Any]],
                                 destylized: bool = False, v0_dir: str = None, progress_tracker=None,
                                 force_comments: bool = False, force_v1: bool = False,
                                 non_regression_prompts_by_app: Dict[str, str] = None) -> Dict[str, Any]:
        if not app_batch_data:
            return {'success': True, 'message': 'No apps to process', 'app_results': []}

        return await process_model_batch(
            self, model_name, app_batch_data, destylized, v0_dir,
            progress_tracker, force_comments, force_v1, non_regression_prompts_by_app
        )
