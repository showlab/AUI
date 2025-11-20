import json
import yaml
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from utils.logging_utils import ts_print
from .prompts.coder_prompts import (
    build_coder_v0_prompt,
    build_coder_v1_failure_prompt,
    build_coder_v1_unsupported_prompt,
)
class Coder:
    def __init__(self, model_client):
        """Code generation agent."""
        self.model_client = model_client
        
    async def generate_initial_website(self, model_name: str, app_name: str, instruction: str,
                                      progress_tracker: Optional[Any] = None,
                                      verbosity: Optional[str] = None,
                                      reasoning_effort: Optional[str] = None) -> str:
        """Generate initial website (enable streaming and progress for GPT-5 series)."""
        prompt = build_coder_v0_prompt(instruction)

        # Enable streaming for GPT-5 / GPT-5.1 to surface liveness
        if model_name in ('gpt5', 'gpt5.1'):
            stream_chars = {'n': 0}
            last_log = {'t': time.time()}

            def _stream_cb(piece: str):
                try:
                    stream_chars['n'] += len(piece or '')
                    now = time.time()
                    if progress_tracker and (now - last_log['t'] >= 5 or stream_chars['n'] < 40):
                        last_log['t'] = now
                        try:
                            progress_tracker.add_timing_info(
                                model_name, app_name,
                                f"{app_name}: 📝 streaming {stream_chars['n']} chars"
                            )
                        except Exception:
                            pass
                except Exception:
                    pass

            v = verbosity if verbosity else "low"
            # GPT-5.1 defaults to reasoning_effort=none for speed; GPT-5 keeps low overhead
            if model_name == 'gpt5.1':
                r = reasoning_effort if reasoning_effort else "none"
            else:
                r = reasoning_effort if reasoning_effort else "low"

            response = await self.model_client.call_coder(
                model_name, prompt,
                verbosity=v,
                reasoning_effort=r,
                stream_callback=_stream_cb
            )
        else:
            response = await self.model_client.call_coder(model_name, prompt)

        html_content = self._extract_html_from_response(response)
        return html_content
    
    async def generate_revised_website(self, model_name: str, app_name: str, 
                           v0_html: str, failed_tasks: List[Dict[str, Any]], 
                           failure_analysis: str = None, apply_destylization: bool = False,
                           v0_dir: str = None, progress_tracker: Optional[Any] = None,
                           verbosity: Optional[str] = None, reasoning_effort: Optional[str] = None,
                           non_regression_contract_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate revised website from failures – integrated analysis and improvement."""
        # Use full initial HTML directly (no summarization)
        
        # Load task descriptions for context
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
        
        # Analyze failure patterns
        failure_categories = {}
        for task in failed_tasks:
            task_id = task.get('task_index', 0)
            task_info = task_descriptions.get(task_id, {})
            category = task_info.get('category', 'unknown')
            
            if category not in failure_categories:
                failure_categories[category] = []
            failure_categories[category].append({
                'task_id': task_id,
                'description': task.get('description', task_info.get('description', 'Unknown')),
                'expected_outcome': task_info.get('expected_outcome', 'Unknown')
            })
        
        # Build concise failure analysis - summarize by category
        detailed_failures = [f"## FAILURE SUMMARY"]
        for category, category_tasks in failure_categories.items():
            detailed_failures.append(f"\n**{category.upper()}** ({len(category_tasks)} tasks): {', '.join([f'#{task["task_id"]}' for task in category_tasks])}")
        
        failure_summary = "\n".join(detailed_failures)
        v0_length = len(v0_html.strip())
        
        # Local models use unbounded retries; cloud models use a capped retry count
        is_local_model = model_name in ['qwen', 'uitars']
        max_retries = float('inf') if is_local_model else 5
        retry_details = []
        
        attempt = 0
        while True:
            try:
                start_time = time.time()
                # Attempt start heartbeat (persistent)
                if progress_tracker:
                    msg = f"{app_name}: ▶️ attempt {attempt + 1} sending request"
                    try:
                        progress_tracker.add_timing_info(model_name, "BATCH", msg)
                        progress_tracker.add_timing_info(model_name, app_name, msg)
                    except Exception:
                        pass

                prompt = build_coder_v1_failure_prompt(
                    app_name=app_name,
                    model_name=model_name,
                    v0_html=v0_html,
                    failed_tasks_len=len(failed_tasks),
                    failure_categories_keys=list(failure_categories.keys()),
                    non_regression_contract_prompt=(non_regression_contract_prompt or ''),
                    failure_analysis=(failure_analysis or failure_summary),
                    apply_destylization=bool(apply_destylization),
                )
                
                # Streaming support for GPT-5 series to surface liveness
                stream_chars = {'n': 0}
                last_log = {'t': time.time()}

                def _stream_cb(piece: str):
                    try:
                        stream_chars['n'] += len(piece or '')
                        now = time.time()
                        if progress_tracker and (now - last_log['t'] >= 5 or stream_chars['n'] < 40):
                            last_log['t'] = now
                            progress_tracker.add_timing_info(
                                model_name, "BATCH",
                                f"{app_name}: 📝 streaming {stream_chars['n']} chars"
                            )
                            try:
                                progress_tracker.update_status(
                                    model_name, app_name,
                                    f"📝 Streaming {stream_chars['n']} chars"
                                )
                            except Exception:
                                pass
                    except Exception:
                        pass

                # GPT-5 and GPT-5.1 both use streaming; 5.1 defaults to reasoning_effort=none
                if model_name == 'gpt5.1':
                    response = await self.model_client.call_coder(
                        model_name, prompt,
                        verbosity=(verbosity or "high"),
                        reasoning_effort=(reasoning_effort or "none"),
                        stream_callback=_stream_cb
                    )
                elif model_name == 'gpt5':
                    response = await self.model_client.call_coder(
                        model_name, prompt,
                        verbosity=(verbosity or "high"),
                        reasoning_effort=(reasoning_effort or "high"),
                        stream_callback=_stream_cb
                    )
                else:
                    response = await self.model_client.call_coder(model_name, prompt)
                html_content = self._extract_html_from_response(response)
                generation_time = time.time() - start_time
                
                html_length = len(html_content.strip())
                success = html_length > 100  # Basic length check
                
                retry_details.append({
                    'attempt': attempt + 1,
                    'generation_time': round(generation_time, 2),
                    'html_length': html_length,
                    'success': success
                })
                
                if success:
                    if progress_tracker:
                        try:
                            progress_tracker.add_timing_info(
                                model_name, "BATCH",
                                f"{app_name}: ✅ attempt {attempt + 1} done in {generation_time:.1f}s, len={html_length}"
                            )
                            progress_tracker.add_timing_info(
                                model_name, app_name,
                                f"{app_name}: ✅ attempt {attempt + 1} done in {generation_time:.1f}s, len={html_length}"
                            )
                        except Exception:
                            pass
                    return {
                        'success': True,
                        'html_content': html_content,
                        'failed_tasks_analyzed': len(failed_tasks),
                        'failure_categories': list(failure_categories.keys()),
                        'attempts': attempt + 1,
                        'html_length': html_length,
                        'retry_details': retry_details
                    }
                else:
                    # For local models, retry on short HTML
                    if is_local_model:
                        attempt += 1
                        # Debug: show actual returned content when HTML is too short
                        short_content = html_content[:50] if html_content else "None"
                        ts_print(f"🔄 {model_name} generated short HTML ({html_length} chars), content: {repr(short_content)}, retrying (attempt {attempt})")
                        if progress_tracker:
                            try:
                                progress_tracker.add_timing_info(
                                    model_name, "BATCH",
                                    f"{app_name}: ❌ attempt {attempt} too short ({html_length} chars), retrying"
                                )
                                progress_tracker.add_timing_info(
                                    model_name, app_name,
                                    f"{app_name}: ❌ attempt {attempt} too short ({html_length} chars), retrying"
                                )
                            except Exception:
                                pass
                        continue
                    else:
                        # Cloud models: record that an outer loop will retry
                        if progress_tracker:
                            try:
                                progress_tracker.add_timing_info(
                                    model_name, "BATCH",
                                    f"{app_name}: ❌ attempt {attempt + 1} too short ({html_length} chars), will retry"
                                )
                                progress_tracker.add_timing_info(
                                    model_name, app_name,
                                    f"{app_name}: ❌ attempt {attempt + 1} too short ({html_length} chars), will retry"
                                )
                            except Exception:
                                pass
                    
            except Exception as e:
                generation_time = time.time() - start_time if 'start_time' in locals() else 0
                retry_details.append({
                    'attempt': attempt + 1,
                    'generation_time': round(generation_time, 2),
                    'error': str(e),
                    'success': False
                })
                if progress_tracker:
                    err = str(e).replace('\n', ' ')[:120]
                    try:
                        progress_tracker.add_timing_info(model_name, "BATCH", f"{app_name}: ❌ attempt {attempt + 1} error: {err}")
                        progress_tracker.add_timing_info(model_name, app_name, f"{app_name}: ❌ attempt {attempt + 1} error: {err}")
                    except Exception:
                        pass
                
                # For local models, retry on all exceptions
                if is_local_model:
                    attempt += 1
                    error_msg = str(e)[:50]
                    ts_print(f"🔄 {model_name} error: {error_msg}..., retrying (attempt {attempt})")
                    if progress_tracker:
                        try:
                            progress_tracker.add_timing_info(model_name, "BATCH", f"{app_name}: 🔄 retrying (attempt {attempt})")
                            progress_tracker.add_timing_info(model_name, app_name, f"{app_name}: 🔄 retrying (attempt {attempt})")
                        except Exception:
                            pass
                    continue
                    
            # For cloud models, respect max_retries
            attempt += 1
            if not is_local_model and attempt >= max_retries:
                break
                
        # All retries failed
        return {
            'success': False,
            'error': f'Failed to generate complete HTML after {max_retries} attempts',
            'html_content': html_content if 'html_content' in locals() else '',
            'attempts': max_retries,
            'retry_details': retry_details
        }
    
    async def generate_unsupported_revision(self, model_name: str, app_name: str, 
                                          v0_html: str, unsupported_tasks: List[Dict], 
                                          task_descriptions: Dict,
                                          non_regression_contract_prompt: Optional[str] = None,
                                          ablate_unsupported_reason: bool = False,
                                          ablate_desc_only: bool = False,
                                          ablate_no_contract: bool = False) -> Dict[str, Any]:
        """Generate revision to support unsupported tasks"""
        
        # Analyze unsupported task patterns
        unsupported_details = []
        for task in unsupported_tasks:
            task_id = task['task_id']
            task_info = task_descriptions.get(task_id, {})
            
            if ablate_desc_only:
                unsupported_details.append(f"""
Task {task_id}: {task_info.get('description', 'Unknown task')}
""")
            elif ablate_unsupported_reason:
                unsupported_details.append(f"""
Task {task_id}: {task_info.get('description', 'Unknown task')}
Expected Outcome: {task_info.get('expected_outcome', 'Unknown')}
Category: {task_info.get('category', 'unknown')}
""")
            else:
                unsupported_details.append(f"""
Task {task_id}: {task_info.get('description', 'Unknown task')}
Expected Outcome: {task_info.get('expected_outcome', 'Unknown')}
Category: {task_info.get('category', 'unknown')}
Reason Unsupported: {task.get('reason', 'Unknown reason')}
""")
        
        unsupported_summary = "\n".join(unsupported_details)
        
        # Use full initial HTML directly (no summarization)

        prompt = build_coder_v1_unsupported_prompt(
            app_name=app_name,
            model_name=model_name,
            v0_html=v0_html,
            unsupported_summary=unsupported_summary,
            non_regression_contract_prompt=(non_regression_contract_prompt or ''),
            ablate_no_contract=ablate_no_contract,
        )

        # Local models use unbounded retries; cloud models use a capped retry count
        is_local_model = model_name in ['qwen', 'uitars']
        max_retries = float('inf') if is_local_model else 5
        
        attempt = 0
        while True:
            start_time = time.time()
            try:
                response = await self.model_client.call_coder(model_name, prompt)
                html_content = self._extract_html_from_response(response)
                generation_time = time.time() - start_time
                
                if len(html_content.strip()) > 100:
                    return {
                        'success': True,
                        'html_content': html_content,
                        'generation_time': generation_time,
                        'attempts': attempt + 1
                    }
                else:
                    # For local models, retry on short HTML
                    if is_local_model:
                        attempt += 1
                        # Debug: show actual returned content when HTML is too short
                        short_content = html_content[:50] if html_content else "None"
                        ts_print(f"🔄 {model_name} unsupported revision generated short HTML ({len(html_content.strip())} chars), content: {repr(short_content)}, retrying (attempt {attempt})")
                        continue
                        
            except Exception as e:
                if is_local_model:
                    attempt += 1
                    error_msg = str(e)[:50]
                    ts_print(f"🔄 {model_name} unsupported revision error: {error_msg}..., retrying (attempt {attempt})")
                    continue
                else:
                    # Cloud models: propagate error directly
                    raise e
                    
            # For cloud models, enforce max_retries
            attempt += 1
            if not is_local_model and attempt >= max_retries:
                break
        
        return {
            'success': False,
            'error': f'Failed to generate valid HTML after {max_retries} attempts'
        }
    
    def _extract_html_from_response(self, response_text: str) -> str:
        """Extract HTML content from model response."""
        # Try to find an explicit HTML code block
        lines = response_text.split('\n')
        html_lines = []
        in_html_block = False
        
        for line in lines:
            # Detect start of HTML code block
            if '```html' in line.lower() or '```' in line and in_html_block == False and '<!DOCTYPE' in response_text:
                in_html_block = True
                continue
            
            # Detect end of HTML code block
            if '```' in line and in_html_block:
                break
                
            # If inside HTML block, collect line
            if in_html_block:
                html_lines.append(line)
            
            # If there is no explicit code fence but HTML appears, start collecting
            if '<!DOCTYPE html>' in line or '<html' in line:
                in_html_block = True
                html_lines.append(line)
        
        # If no block was found, attempt coarse extraction from raw response; otherwise return empty to trigger retries
        if not html_lines:
            if '<!DOCTYPE html>' in response_text or '<html' in response_text:
                start_idx = response_text.find('<!DOCTYPE html>')
                if start_idx == -1:
                    start_idx = response_text.find('<html')
                end_idx = response_text.rfind('</html>')
                if start_idx != -1 and end_idx != -1:
                    return response_text[start_idx:end_idx + 7]
            return ""
        
        html_content = '\n'.join(html_lines)
        
        # Verify HTML is non-empty
        if not html_content.strip():
            return ""  # Return empty so caller can detect and retry/fail
        
        # Ensure basic HTML structure exists
        if '<!DOCTYPE html>' not in html_content and '<html' not in html_content:
            html_content = ""
        
        return html_content
    
     
    def load_app_instruction(self, app_name: str) -> str:
        """Load app instruction prompt from YAML."""
        instruction_path = Path(f"examples/{app_name}.yaml")
        with open(instruction_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('prompt', '')
    
    def save_website(self, html_content: str, app_name: str, model_name: str, phase: str = "initial", base_dir: str = "websites") -> str:
        """Save website file to disk."""
        if phase == "initial":
            website_dir = Path(f"{base_dir}/{app_name}/{model_name}")
        else:  # revised
            website_dir = Path(f"experiments/{phase}/{app_name}/{model_name}/revised_website")
        
        website_dir.mkdir(parents=True, exist_ok=True)
        
        html_path = website_dir / "index.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(html_path)
