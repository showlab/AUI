import json
import tempfile
import asyncio
from typing import List, Dict, Any, Tuple
from pathlib import Path
from abc import ABC, abstractmethod

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.browser import BrowserController

# Global browser pool to prevent resource exhaustion
_browser_pool = []
_browser_semaphore = asyncio.Semaphore(5)  # Max 5 concurrent browsers
_pool_lock = asyncio.Lock()

class BaseCommenter(ABC):
    def __init__(self, model_client):
        """Base class for analyzing UI issues from failed CUA trajectories."""
        self.model_client = model_client
        # Map logical model names to concrete VLM configs
        self.model_mapping = {
            'qwen': 'qwen2.5-vl-72b'
        }
        
    def _get_actual_model_name(self, model_name: str) -> str:
        """Resolve the actual model name used by the client."""
        return self.model_mapping.get(model_name, model_name)
    
    async def _safe_capture_screenshot(self, html_content: str, timeout_seconds: int = 30) -> tuple[str, tuple[int, int]]:
        """Safely capture screenshot with timeout; no fallback image on failure"""
        return await asyncio.wait_for(
            self._capture_version_screenshot(html_content), 
            timeout=timeout_seconds
        )
    
    async def _get_browser_from_pool(self):
        """Get browser from pool with strict resource limits"""
        async with _pool_lock:
            if _browser_pool:
                return _browser_pool.pop()
        
        # No available browsers, create new one with semaphore limit
        async with _browser_semaphore:
            browser = BrowserController(headless=True, width=1280, height=1024)
            await browser.start()
            return browser
    
    async def _return_browser_to_pool(self, browser):
        """Return browser to pool or close if pool is full"""
        async with _pool_lock:
            if len(_browser_pool) < 3:  # Keep max 3 browsers in pool
                _browser_pool.append(browser)
            else:
                try:
                    await browser.close()
                except:
                    pass  # Ignore cleanup errors
    
    async def _capture_version_screenshot(self, html_content: str) -> tuple[str, tuple[int, int]]:
        """Capture screenshot for HTML version using a browser pool to avoid resource exhaustion.
        
        Returns:
            tuple: (screenshot_base64, (width, height))
        """
        browser = await self._get_browser_from_pool()
        
        try:
            # Create a temporary HTML file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
                f.write(html_content)
                temp_html_path = f.name
            
            # Load page
            await browser.navigate_to(f"file://{temp_html_path}")
            await asyncio.sleep(0.5)  # Short wait to allow layout
            
            # Get actual page size
            page_size = await browser.page.evaluate("""() => {
                return {
                    width: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth),
                    height: Math.max(document.documentElement.scrollHeight, document.body.scrollHeight)
                }
            }""")
            
            # Take screenshot
            screenshot_base64 = await browser.screenshot()
            
            # Clean up temporary file
            Path(temp_html_path).unlink()
            
            return screenshot_base64, (page_size['width'], page_size['height'])
            
        finally:
            await self._return_browser_to_pool(browser)
    
    @abstractmethod
    def _prepare_analysis_inputs(self, storyboard_path: str, html_content: str, website_screenshot: str, width: int, height: int) -> tuple[str, List[str]]:
        """Prepare analysis inputs – subclasses implement concrete logic.
        
        Returns:
            tuple: (prompt, screenshot_inputs)
        """
        pass
    
    async def analyze_single_failure(self, storyboard_path: str, html_content: str, model_name: str = "gpt5") -> str:
        """Analyze UI issues for a single failed CUA trajectory.
        
        Args:
            storyboard_path: Path to storyboard image.
            html_content: Website HTML content.
            model_name: Model used for analysis (gpt5, gpt4o, qwen, etc.).
            
        Returns:
            str: Analysis of UI design issues that caused the failure.
        """
        try:
            # Check storyboard presence
            if not Path(storyboard_path).exists():
                return "Storyboard image not found - cannot analyze failure"
            
            # Capture website screenshot for comparison
            website_screenshot, (width, height) = await self._safe_capture_screenshot(html_content)
            
            # Let subclass prepare concrete analysis inputs
            prompt, screenshot_inputs = self._prepare_analysis_inputs(
                storyboard_path, html_content, website_screenshot, width, height
            )
            
            # Use configured model for analysis (typically VLM such as GPT-5 or GPT-4o)
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    # Run visual analysis with the resolved model
                    actual_model = self._get_actual_model_name(model_name)
                    response = await self.model_client.call_commenter(actual_model, prompt, screenshot_inputs)
                    if response and len(response.strip()) > 30:
                        return response.strip()
                except Exception as e:
                    if attempt == max_retries - 1:
                        return f"Failed to analyze failure after {max_retries} attempts: {str(e)}"
                    await asyncio.sleep(1)  # Brief pause between retries
                    continue
            
            return "Unable to analyze failure - no valid response received"
            
        except Exception as e:
            return f"Error analyzing failure trajectory: {str(e)}"
