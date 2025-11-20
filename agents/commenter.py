import base64
from typing import List, Tuple
from pathlib import Path

from .base_commenter import BaseCommenter
from .prompts.commenter_prompts import build_storyboard_prompt

class Commenter(BaseCommenter):
    def _prepare_analysis_inputs(self, storyboard_path: str, html_content: str, website_screenshot: str, width: int, height: int) -> Tuple[str, List[str]]:
        """Prepare analysis inputs using storyboard image."""
        # Load storyboard image
        with open(storyboard_path, 'rb') as f:
            storyboard_image = base64.b64encode(f.read()).decode('utf-8')
        
        # Build analysis prompt (expects structured JSON output)
        prompt = build_storyboard_prompt(width, height)

        return prompt, [website_screenshot, storyboard_image]
