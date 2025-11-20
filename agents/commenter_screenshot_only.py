import base64
import json
from typing import List, Tuple
from pathlib import Path

from .base_commenter import BaseCommenter
from .prompts.commenter_prompts import build_screenshot_only_prompt

class CommenterScreenshotOnly(BaseCommenter):
    def _load_step_screenshots(self, trajectory_dir: str) -> List[str]:
        """Load step screenshots from trajectory directory (based on trajectory.json)."""
        trajectory_path = Path(trajectory_dir)
        step_screenshots = []
        
        # First read trajectory.json to recover the actual number of steps
        trajectory_file = trajectory_path / "trajectory.json"
        actual_steps = 0
        if trajectory_file.exists():
            try:
                with open(trajectory_file, 'r', encoding='utf-8') as f:
                    trajectory_data = json.load(f)
                actual_steps = len(trajectory_data)
            except Exception:
                pass
        
        # If trajectory.json is missing, fall back to checking existing files
        if actual_steps == 0:
            for step_num in range(1, 21):  # Fallback scan up to 20 steps
                step_file = trajectory_path / f"step_{step_num}.png"
                if step_file.exists():
                    actual_steps = step_num
                else:
                    break
        
        # Load existing step screenshots starting from step_1 (skip step_0)
        for step_num in range(1, actual_steps + 1):
            step_file = trajectory_path / f"step_{step_num}.png"
            if step_file.exists():
                with open(step_file, 'rb') as f:
                    screenshot_base64 = base64.b64encode(f.read()).decode('utf-8')
                    step_screenshots.append(screenshot_base64)
        
        return step_screenshots
    
    def _prepare_analysis_inputs(self, storyboard_path: str, html_content: str, website_screenshot: str, width: int, height: int) -> Tuple[str, List[str]]:
        """Prepare analysis inputs using raw step screenshots."""
        # Derive trajectory directory from storyboard path
        storyboard_path_obj = Path(storyboard_path)
        trajectory_dir = storyboard_path_obj.parent
        
        # Load step screenshots for the actual number of steps
        step_screenshots = self._load_step_screenshots(str(trajectory_dir))
        
        # Filter out empty screenshots
        valid_screenshots = [s for s in step_screenshots if s]
        if not valid_screenshots:
            raise ValueError("No valid step screenshots found")
        
        # Build analysis prompt (expects structured JSON output)
        prompt = build_screenshot_only_prompt(width, height, len(valid_screenshots))

        return prompt, [website_screenshot] + valid_screenshots
