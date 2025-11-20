import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.browser import BrowserController
from utils.action_parser import parse_action_to_structure_output

class BaseCUAPolicy:
    def __init__(self, model_client, model_name: str = "uitars", max_steps: int = 10):
        """Base CUA Policy with state-change termination logic"""
        self.model_client = model_client
        self.model_name = model_name
        self.max_steps = max_steps
        self.display_width = 1280
        self.display_height = 720
        # State change tracking
        self.consecutive_no_change_steps = 0
        self.max_no_change_steps = 3
        
    def _compare_states(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> bool:
        """Compare two page states; any change is meaningful for CUA progress"""
        if not state1 and not state2:
            return False
        if not state1 or not state2:
            return True
        # Direct comparison includes visibility, scroll, classes, dataset, etc.
        return state1 != state2
    
    def _should_terminate_due_to_stuck(self, current_state: Dict[str, Any], previous_state: Dict[str, Any]) -> bool:
        """Check if should terminate due to being stuck (3 consecutive steps with no state change)"""
        if self._compare_states(current_state, previous_state):
            # State changed, reset counter
            self.consecutive_no_change_steps = 0
            return False
        else:
            # No state change, increment counter
            self.consecutive_no_change_steps += 1
            return self.consecutive_no_change_steps >= self.max_no_change_steps
    
    async def execute_task(self, app_name: str, model_name: str, website_url: str, 
                          task: Dict[str, Any], completion_rule: str,
                          save_dir: Optional[str] = None) -> Dict[str, Any]:
        """Execute single task with state-change-based termination"""
        
        try:
            trajectory = []
            browser = BrowserController(headless=True)
            
            await browser.start()
            await browser.navigate_to(website_url)
            await browser.inject_state_monitor_script()
            # Wait a couple of frames to stabilize the initial view
            try:
                await browser.wait_for_animation_frames(2)
            except Exception:
                pass
            
            if not save_dir:
                raise ValueError("save_dir is required for CUA Policy execution")
            
            # Initial state
            initial_state = await browser.get_page_state()
            previous_state = initial_state.copy()
            
            task_description = task.get('description', '')
            success_criteria = task.get('success_criteria', '')
            
            # Reset state change tracking
            self.consecutive_no_change_steps = 0
            
            # Execute task steps
            for step in range(1, self.max_steps + 1):
                current_screenshot_b64 = await browser.viewport_screenshot()
                
                prompt = self._build_computer_use_prompt(
                    task_description, 
                    success_criteria,
                    trajectory,
                    step
                )
                
                response = await self._get_computer_use_action(prompt, current_screenshot_b64)
                # Save raw LLM response if provided by policy (e.g., operator)
                if isinstance(response, dict) and 'raw' in response and response['raw']:
                    raw_path = Path(save_dir) / f"raw_step_{step}.json"
                    with open(raw_path, 'w', encoding='utf-8') as rf:
                        rf.write(response['raw'])
                
                if not response:
                    print(f"⚠️ Step {step}: No action returned")
                    break
                
                action = response.get("action") if isinstance(response, dict) and "action" in response else response
                thought = response.get("thought", "") if isinstance(response, dict) else ""
                reasoning_source = response.get("reasoning_source", "") if isinstance(response, dict) else ""
                
                # Check for explicit termination
                if action.get('action') == 'terminate':
                    status = action.get('status', 'success')
                    print(f"     Task terminated with status: {status}")
                    
                    final_state = await browser.get_page_state()
                    
                    screenshot_path = Path(save_dir) / f"step_{step}.png"
                    await browser.viewport_screenshot(str(screenshot_path))
                    screenshot_reference = str(screenshot_path)
                    
                    trajectory.append({
                        'step': step,
                        'screenshot': screenshot_reference,
                        'thought': thought,
                        'reasoning_source': reasoning_source,
                        'action': action,
                        'result': {"success": status == 'success', "message": "Task terminated"},
                        'state': final_state
                    })
                    
                    rule_str = completion_rule
                    completed = self._check_task_completion(rule_str, final_state)
                    
                    await browser.close()
                    return {
                        'success': True,
                        'completed': completed,
                        'steps': step,
                        'trajectory': trajectory,
                        'termination_reason': 'explicit'
                    }
                
                # Execute action
                print(f"     Step {step}: {action['action']} {action.get('coordinate', '')}")
                result = await self._execute_computer_use_action(browser, action)
                
                # Get new state after action
                new_state = await browser.get_page_state()
                
                # Check if stuck (3 consecutive steps with no state change)
                if self._should_terminate_due_to_stuck(new_state, previous_state):
                    print(f"     Terminating due to being stuck (no state change for {self.max_no_change_steps} steps)")
                    
                    screenshot_path = Path(save_dir) / f"step_{step}.png"
                    await browser.viewport_screenshot(str(screenshot_path))
                    screenshot_reference = str(screenshot_path)
                    
                    trajectory.append({
                        'step': step,
                        'screenshot': screenshot_reference,
                        'thought': thought,
                        'action': action,
                        'result': result,
                        'state': new_state
                    })
                    
                    rule_str = completion_rule
                    completed = self._check_task_completion(rule_str, new_state)
                    
                    await browser.close()
                    return {
                        'success': True,
                        'completed': completed,
                        'steps': step,
                        'trajectory': trajectory,
                        'termination_reason': 'stuck'
                    }
                
                # Screenshot and record trajectory
                screenshot_path = Path(save_dir) / f"step_{step}.png"
                await browser.viewport_screenshot(str(screenshot_path))
                screenshot_reference = str(screenshot_path)
                
                trajectory.append({
                    'step': step,
                    'screenshot': screenshot_reference,
                    'thought': thought,
                    'reasoning_source': reasoning_source,
                    'action': action,
                    'result': result,
                    'state': new_state
                })
                
                if not result.get('success'):
                    print(f"       ❌ Failed: {result.get('error', 'Unknown error')}")
                    if step >= 2:  
                        break
                else:
                    print(f"       ✅ Success: {result.get('message', 'Action completed')}")
                
                # Immediately check whether completion rule is satisfied (early stop)
                rule_str = completion_rule
                try:
                    if self._check_task_completion(rule_str, new_state):
                        await browser.close()
                        return {
                            'success': True,
                            'completed': True,
                            'steps': step,
                            'trajectory': trajectory,
                            'termination_reason': 'rule_satisfied'
                        }
                except Exception:
                    pass

                # Update previous state for next comparison
                previous_state = new_state.copy()
                
                await asyncio.sleep(1.5)
            
            # Reached max steps
            final_state = await browser.get_page_state()
            rule_str = completion_rule
            final_completed = self._check_task_completion(rule_str, final_state)
            
            await browser.close()
            return {
                'success': True,
                'completed': final_completed,
                'steps': self.max_steps,
                'trajectory': trajectory,
                'termination_reason': 'max_steps'
            }
            
        except Exception as e:
            try:
                if 'browser' in locals():
                    await browser.close()
            except:
                pass
            
            return {
                'success': False,
                'error': str(e),
                'completed': False,
                'steps': 0,
                'trajectory': [],
                'termination_reason': 'error'
            }
    
    def _build_computer_use_prompt(self, task_description: str, success_criteria: str, 
                                   trajectory: List[Dict], current_step: int) -> str:
        """Build computer use prompt - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _build_computer_use_prompt")
    
    async def _get_computer_use_action(self, prompt: str, screenshot: str) -> Dict[str, Any]:
        """Get computer use action - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _get_computer_use_action")
    
    def _convert_parsed_to_internal(self, parsed_action: Dict[str, Any]) -> Dict[str, Any]:
        """Convert parsed action - can be overridden by subclasses"""
        action_type = parsed_action.get("action_type")
        action_inputs = parsed_action.get("action_inputs", {})
        
        if action_type == "click" or action_type == "left_single":
            start_box = action_inputs.get("start_box")
            if start_box:
                coords = eval(start_box)
                if len(coords) == 4:
                    x = int(((coords[0] + coords[2]) / 2) * self.display_width)
                    y = int(((coords[1] + coords[3]) / 2) * self.display_height)
                else:
                    x = int(coords[0] * self.display_width)
                    y = int(coords[1] * self.display_height)
                return {"action": "left_click", "coordinate": [x, y]}
        
        elif action_type == "left_double":
            start_box = action_inputs.get("start_box")
            if start_box:
                coords = eval(start_box)
                if len(coords) == 4:
                    x = int(((coords[0] + coords[2]) / 2) * self.display_width)
                    y = int(((coords[1] + coords[3]) / 2) * self.display_height)
                else:
                    x = int(coords[0] * self.display_width)
                    y = int(coords[1] * self.display_height)
                return {"action": "double_click", "coordinate": [x, y]}
        
        elif action_type == "right_single":
            start_box = action_inputs.get("start_box")
            if start_box:
                coords = eval(start_box)
                if len(coords) == 4:
                    x = int(((coords[0] + coords[2]) / 2) * self.display_width)
                    y = int(((coords[1] + coords[3]) / 2) * self.display_height)
                else:
                    x = int(coords[0] * self.display_width)
                    y = int(coords[1] * self.display_height)
                return {"action": "right_click", "coordinate": [x, y]}
        
        elif action_type == "type":
            content = action_inputs.get("content", "")
            return {"action": "type", "text": content}
        
        elif action_type == "scroll":
            start_box = action_inputs.get("start_box")
            direction = action_inputs.get("direction", "down")
            pixels = -300 if direction == "down" else 300
            if start_box:
                coords = eval(start_box)
                if len(coords) == 4:
                    x = int(((coords[0] + coords[2]) / 2) * self.display_width)
                    y = int(((coords[1] + coords[3]) / 2) * self.display_height)
                else:
                    x = int(coords[0] * self.display_width)
                    y = int(coords[1] * self.display_height)
                return {"action": "scroll", "coordinate": [x, y], "pixels": pixels}
            else:
                return {"action": "scroll", "pixels": pixels}
        
        elif action_type == "finished":
            content = action_inputs.get("content", "success")
            status = "success" if "success" in content.lower() else "failure"
            return {"action": "terminate", "status": status}
        
        elif action_type == "hotkey":
            key = action_inputs.get("key", "")
            keys = key.split() if key else []
            return {"action": "key", "keys": keys}
        
        elif action_type == "drag":
            start_box = action_inputs.get("start_box")
            end_box = action_inputs.get("end_box")
            if start_box and end_box:
                end_coords = eval(end_box)
                if len(end_coords) == 4:
                    x = int(((end_coords[0] + end_coords[2]) / 2) * self.display_width)
                    y = int(((end_coords[1] + end_coords[3]) / 2) * self.display_height)
                else:
                    x = int(end_coords[0] * self.display_width)
                    y = int(end_coords[1] * self.display_height)
                return {"action": "left_click_drag", "coordinate": [x, y]}
        
        elif action_type == "wait":
            return {"action": "wait", "time": 5}
        
        print(f"❌ Unsupported action type: {action_type}")
        return None
    
    async def _execute_computer_use_action(self, browser, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute computer use action on browser"""
        action_type = action.get('action')
        
        if action_type == 'left_click':
            coord = action.get('coordinate', [])
            if len(coord) >= 2:
                return await browser.click_at_coordinates(coord[0], coord[1])
            else:
                return {"success": False, "error": "Invalid coordinates"}
                
        elif action_type == 'type':
            text = action.get('text', '')
            return await browser.type_text(text)
            
        elif action_type == 'mouse_move':
            coord = action.get('coordinate', [])
            if len(coord) >= 2:
                return await browser.move_to_coordinates(coord[0], coord[1])
            else:
                return {"success": False, "error": "Invalid coordinates"}
        
        elif action_type == 'drag':
            # Expected structure: {'from': [x1,y1], 'to': [x2,y2]}
            start = action.get('from') or action.get('start')
            end = action.get('to') or action.get('end')
            if start and end and len(start) >= 2 and len(end) >= 2:
                return await browser.drag_from_to(int(start[0]), int(start[1]), int(end[0]), int(end[1]))
            else:
                return {"success": False, "error": "Invalid drag coordinates"}
                
        elif action_type == 'scroll':
            # Support precise scrolling with pixel deltas and optional anchor coordinate
            if 'pixels_x' in action or 'pixels_y' in action:
                px = int(action.get('pixels_x', 0) or 0)
                py = int(action.get('pixels_y', 0) or 0)
                coord = action.get('coordinate', [])
                if len(coord) >= 2:
                    # Move mouse to anchor before scrolling
                    await browser.move_to_coordinates(coord[0], coord[1])
                return await browser.scroll_by(px, py)
            else:
                pixels = action.get('pixels', 0)
                direction = 'down' if pixels < 0 else 'up'
                return await browser.scroll(direction)
            
        elif action_type == 'key':
            keys = action.get('keys', [])
            return await browser.press_keys(keys)
            
        elif action_type == 'wait':
            time_sec = action.get('time', 1)
            await asyncio.sleep(time_sec)
            return {"success": True, "message": f"Waited {time_sec} seconds"}
            
        elif action_type == 'double_click':
            coord = action.get('coordinate', [])
            if len(coord) >= 2:
                return await browser.double_click_at_coordinates(coord[0], coord[1])
            else:
                return {"success": False, "error": "Invalid coordinates"}
                
        elif action_type == 'right_click':
            coord = action.get('coordinate', [])
            if len(coord) >= 2:
                return await browser.right_click_at_coordinates(coord[0], coord[1])
            else:
                return {"success": False, "error": "Invalid coordinates"}
        
        elif action_type == 'left_click_drag':
            coord = action.get('coordinate', [])
            if len(coord) >= 2:
                return await browser.drag_to_coordinates(coord[0], coord[1])
            else:
                return {"success": False, "error": "Invalid coordinates"}

        elif action_type == 'screenshot':
            # No-op: model already sees screenshots every round; just acknowledge
            await asyncio.sleep(0.1)
            return {"success": True, "message": "Screenshot acknowledged"}
                
        elif action_type == 'terminate':
            return {"success": True, "message": "Task terminated"}
            
        else:
            return {"success": False, "error": f"Unknown action: {action_type}"}
    
    def _check_task_completion(self, completion_rule: str, page_state: Dict[str, Any]) -> bool:
        """Check task completion using judge"""
        if not completion_rule:
            return False
        
        from .judge import Judge
        judge = Judge(None)
        return judge.evaluate_task_completion(completion_rule, page_state)
