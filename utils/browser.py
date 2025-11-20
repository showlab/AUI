import asyncio
import json
from typing import Dict, Any, List, Optional
from playwright.async_api import async_playwright


class BrowserController:
    def __init__(self, headless: bool = True, width: int = 1280, height: int = 720):
        """Minimal browser controller for AUI experiments."""
        self.headless = headless
        self.width = width
        self.height = height
        self.playwright = None
        self.browser = None
        self.page = None
        
    async def start(self):
        """Launch browser."""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=['--no-sandbox', '--disable-dev-shm-usage']
        )
        context = await self.browser.new_context(
            viewport={'width': self.width, 'height': self.height}
        )
        self.page = await context.new_page()
        
    async def close(self):
        """Close browser and stop Playwright."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    async def navigate_to(self, url: str):
        """Navigate to the given URL."""
        await self.page.goto(url, wait_until='domcontentloaded', timeout=60000)
    
    async def screenshot(self, path: Optional[str] = None, full_page: bool = True) -> str:
        """Capture screenshot (full page or viewport)."""
        if path:
            await self.page.screenshot(path=path, full_page=full_page)
            return path
        else:
            # Return screenshot as base64-encoded string
            screenshot_bytes = await self.page.screenshot(full_page=full_page)
            import base64
            return base64.b64encode(screenshot_bytes).decode()
    
    async def viewport_screenshot(self, path: Optional[str] = None) -> str:
        """Capture viewport-only screenshot (used by CUA)."""
        return await self.screenshot(path, full_page=False)
    
    async def click_at_coordinates(self, x: int, y: int):
        """Click at the given coordinates."""
        await self.page.mouse.click(x, y)
        await asyncio.sleep(0.5)
        return {"success": True, "message": "Clicked"}
    
    async def type_text(self, text: str):
        """Type text into the current focused element."""
        await self.page.keyboard.type(text)
        await asyncio.sleep(0.5)
        return {"success": True, "message": f"Typed: {text}"}
    
    async def scroll(self, direction: str = "down"):
        """Scroll the page up or down."""
        if direction.lower() == "up":
            await self.page.evaluate("window.scrollBy(0, -500)")
        else:
            await self.page.evaluate("window.scrollBy(0, 500)")
        await asyncio.sleep(0.5)
        return {"success": True, "message": f"Scrolled {direction}"}

    async def scroll_by(self, delta_x: int = 0, delta_y: int = 0):
        """Scroll by pixels, supporting horizontal and vertical movement."""
        # Use native wheel to avoid evaluate arg mismatch issues
        await self.page.mouse.wheel(delta_x, delta_y)
        await asyncio.sleep(0.5)
        return {"success": True, "message": f"Scrolled by dx={delta_x}, dy={delta_y}"}

    async def scroll_to_coordinates(self, x: int, y: int, direction: str = "down", pixels: int = 500):
        """Scroll anchored at a given coordinate."""
        scroll_delta = -pixels if direction.lower() == "up" else pixels
        # Move to anchor, then wheel with deltaY
        await self.page.mouse.move(x, y)
        await self.page.mouse.wheel(0, scroll_delta)
        await asyncio.sleep(0.5)
    
    async def double_click_at_coordinates(self, x: int, y: int):
        """Double-click at the given coordinates."""
        await self.page.mouse.dblclick(x, y)
        await asyncio.sleep(0.5)
        return {"success": True, "message": "Double clicked"}
    
    async def right_click_at_coordinates(self, x: int, y: int):
        """Right-click at the given coordinates."""
        await self.page.mouse.click(x, y, button='right')
        await asyncio.sleep(0.5)
        return {"success": True, "message": "Right clicked"}
    
    async def move_to_coordinates(self, x: int, y: int):
        """Move mouse to the given coordinates."""
        await self.page.mouse.move(x, y)
        await asyncio.sleep(0.5)
        return {"success": True, "message": "Mouse moved"}
    
    def _map_key_name(self, key: str) -> str:
        """Map key names to Playwright-compatible names"""
        key_mapping = {
            # Arrow keys
            'arrowleft': 'ArrowLeft',
            'arrowright': 'ArrowRight', 
            'arrowup': 'ArrowUp',
            'arrowdown': 'ArrowDown',
            'left': 'ArrowLeft',
            'right': 'ArrowRight',
            'up': 'ArrowUp',
            'down': 'ArrowDown',
            # Common keys
            'space': 'Space',
            'enter': 'Enter',
            'return': 'Enter',
            'tab': 'Tab',
            'escape': 'Escape',
            'backspace': 'Backspace',
            'delete': 'Delete',
            'shift': 'Shift',
            'ctrl': 'Control',
            'control': 'Control',
            'alt': 'Alt',
            'meta': 'Meta',
            'cmd': 'Meta',
            'home': 'Home',
            'end': 'End',
            'pageup': 'PageUp',
            'pagedown': 'PageDown',
            'insert': 'Insert'
        }
        # Map function keys f1..f12
        lk = key.lower()
        if lk.startswith('f') and lk[1:].isdigit():
            n = int(lk[1:])
            if 1 <= n <= 12:
                return f"F{n}"
        return key_mapping.get(lk, key)

    async def press_keys(self, keys: list):
        """Press a combination of keys."""
        if not keys:
            return {"success": False, "error": "No keys provided"}
        
        # Map all key names to Playwright-compatible names
        mapped_keys = [self._map_key_name(key) for key in keys]
        
        # Press all modifier keys first
        for key in mapped_keys[:-1]:
            await self.page.keyboard.down(key)
        
        # Press the last key
        await self.page.keyboard.press(mapped_keys[-1])
        
        # Release modifier keys
        for key in reversed(mapped_keys[:-1]):
            await self.page.keyboard.up(key)
        
        await asyncio.sleep(0.5)
        return {"success": True, "message": f"Pressed keys: {' + '.join(keys)}"}
    
    async def drag_to_coordinates(self, x: int, y: int):
        """Drag to the given coordinates (mouse must already be down)."""
        await self.page.mouse.move(x, y)
        await self.page.mouse.up()
        await asyncio.sleep(0.5)
        return {"success": True, "message": "Dragged to coordinates"}

    async def mouse_down_at(self, x: int, y: int):
        """Press mouse left button at given coordinates."""
        await self.page.mouse.move(x, y)
        await self.page.mouse.down()
        await asyncio.sleep(0.2)
        return {"success": True, "message": "Mouse down"}

    async def mouse_up(self):
        """Release mouse left button."""
        await self.page.mouse.up()
        await asyncio.sleep(0.2)
        return {"success": True, "message": "Mouse up"}

    async def drag_from_to(self, x1: int, y1: int, x2: int, y2: int):
        """Press at (x1, y1), drag to (x2, y2), then release."""
        await self.page.mouse.move(x1, y1)
        await self.page.mouse.down()
        await self.page.mouse.move(x2, y2)
        await self.page.mouse.up()
        await asyncio.sleep(0.5)
        return {"success": True, "message": f"Dragged from ({x1},{y1}) to ({x2},{y2})"}
    
    async def inject_state_monitor_script(self):
        """Inject state-monitoring script into the page."""
        script = """
        window.AUIStateMonitor = {
            getState: function() {
                const state = {};
                // Global page context useful for detecting progress
                try {
                    const vv = window.visualViewport || {};
                    const se = document.scrollingElement || document.documentElement || document.body;
                    state.__meta_viewport_width = window.innerWidth;
                    state.__meta_viewport_height = window.innerHeight;
                    state.__meta_device_pixel_ratio = window.devicePixelRatio || 1;
                    state.__meta_visual_scale = vv.scale || 1;
                    state.__meta_scroll_top = se.scrollTop || 0;
                    state.__meta_scroll_height = se.scrollHeight || 0;
                    state.__meta_scroll_left = se.scrollLeft || 0;
                    state.__meta_scroll_width = se.scrollWidth || 0;
                    state.__meta_location_hash = location.hash || '';
                    state.__meta_location_path = location.pathname || '';
                    state.__meta_location_search = location.search || '';
                    state.__meta_document_title = document.title || '';
                    const ae = document.activeElement;
                    state.__meta_active_element_id = (ae && ae.id) ? ae.id : '';
                } catch (e) {}
                
                // Extract text for all elements that have an ID
                const elementsWithId = document.querySelectorAll('[id]');
                elementsWithId.forEach(elem => {
                    if (elem.id) {
                        state[elem.id] = elem.textContent.trim();
                        
                        // Extract values for input-like elements
                        if (elem.tagName === 'INPUT' || elem.tagName === 'TEXTAREA' || elem.tagName === 'SELECT') {
                            if (elem.type === 'checkbox' || elem.type === 'radio') {
                                state[elem.id] = elem.checked;
                            } else {
                                state[elem.id] = elem.value;
                            }
                        }
                        
                        // Record visibility
                        try {
                            const cs = getComputedStyle(elem);
                            state[elem.id + '_visible'] = !elem.hidden && cs.display !== 'none' && cs.visibility !== 'hidden' && cs.opacity !== '0';
                        } catch (e) {
                            state[elem.id + '_visible'] = !elem.hidden;
                        }
                        
                        // Record class and data-* attributes to capture state changes
                        try { state[elem.id + '_class'] = elem.className || ''; } catch (e) {}
                        try { state[elem.id + '_data'] = Object.assign({}, elem.dataset || {}); } catch (e) {}
                        // Record aria-* attributes as observable state
                        try {
                            const aria = {};
                            if (elem.attributes) {
                                for (let i = 0; i < elem.attributes.length; i++) {
                                    const attr = elem.attributes[i];
                                    if (attr && attr.name && attr.name.startsWith('aria-')) {
                                        aria[attr.name.substring(5)] = attr.value;
                                    }
                                }
                            }
                            state[elem.id + '_aria'] = aria;
                        } catch (e) {}
                        // Record a small subset of common HTML attributes for rule evaluation
                        try {
                            const attr = {};
                            const names = ['href','src','download','role','type','value'];
                            for (const n of names) {
                                try {
                                    const v = elem.getAttribute(n);
                                    if (v !== null) attr[n] = v;
                                } catch (e2) {}
                            }
                            state[elem.id + '_attr'] = attr;
                        } catch (e) {}
                    }
                });
                
                // Additionally collect elements without IDs but with important classes
                const importantClasses = ['.result', '.output', '.score', '.status', '.message', 
                                        '.timer', '.color-word', '.color-button'];
                importantClasses.forEach(selector => {
                    const elements = document.querySelectorAll(selector);
                    elements.forEach((elem, index) => {
                        const key = selector.replace('.', '') + (index > 0 ? `_${index}` : '');
                        state[key] = elem.textContent.trim();
                        
                        // For input elements, also extract values
                        if (elem.tagName === 'INPUT' || elem.tagName === 'TEXTAREA' || elem.tagName === 'SELECT') {
                            if (elem.type === 'checkbox' || elem.type === 'radio') {
                                state[key] = elem.checked;
                            } else {
                                state[key] = elem.value;
                            }
                        }
                        // Record class to capture part of the visual state
                        try { state[key + '_class'] = elem.className || ''; } catch (e) {}
                    });
                });
                
                // Collect generic input values as a backup
                const inputs = document.querySelectorAll('input, textarea, select');
                inputs.forEach((input, index) => {
                    if (!input.id) {
                        const key = input.name || `input_${index}`;
                        if (input.type === 'checkbox' || input.type === 'radio') {
                            state[key] = input.checked;
                        } else {
                            state[key] = input.value;
                        }
                    }
                });
                
                return state;
            }
        };
        """
        await self.page.evaluate(script)
    
    async def get_page_state(self) -> Dict[str, Any]:
        """Return current page state from injected monitor."""
        state = await self.page.evaluate("window.AUIStateMonitor.getState()")
        return state
    
    async def get_page_content(self) -> str:
        """Get full page HTML content."""
        return await self.page.content()
    
    async def get_page_info(self) -> Dict[str, Any]:
        """Get basic page info (URL, title, readyState)."""
        return {
            "url": self.page.url,
            "title": await self.page.title(),
            "ready_state": await self.page.evaluate("document.readyState")
        }
