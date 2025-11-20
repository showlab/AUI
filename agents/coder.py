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
        """代码生成Agent"""
        self.model_client = model_client
        
    async def generate_initial_website(self, model_name: str, app_name: str, instruction: str,
                                      progress_tracker: Optional[Any] = None,
                                      verbosity: Optional[str] = None,
                                      reasoning_effort: Optional[str] = None) -> str:
        """生成初始网站（对GPT-5系列启用streaming并打印进度）"""
        prompt = build_coder_v0_prompt(instruction)

        # 对 GPT-5 / GPT-5.1 启用 streaming 以便实时进度输出
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
            # GPT-5.1 默认关闭 reasoning effort 以减速；GPT-5 保持低开销
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
        """基于失败生成修订版网站 - 综合分析和改进"""
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
        
        # 本地模型使用无限重试，云端模型限制重试次数
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
                
                # Streaming support for GPT-5 系列 to surface liveness
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

                # GPT-5 与 GPT-5.1 均使用 streaming；5.1 默认 reasoning_effort=none
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
                    # 对于本地模型，短HTML也要重试
                    if is_local_model:
                        attempt += 1
                        # 调试：显示实际返回的内容
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
                        # 云端模型：记录将重试（由外层重试控制）
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
                
                # 对于本地模型，所有异常都重试
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
                    
            # 云端模型检查重试次数
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

        # 本地模型使用无限重试，云端模型限制重试次数
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
                    # 对于本地模型，短HTML也要重试
                    if is_local_model:
                        attempt += 1
                        # 调试：显示实际返回的内容
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
                    # 云端模型直接抛出异常
                    raise e
                    
            # 云端模型检查重试次数
            attempt += 1
            if not is_local_model and attempt >= max_retries:
                break
        
        return {
            'success': False,
            'error': f'Failed to generate valid HTML after {max_retries} attempts'
        }
    
    def _extract_html_from_response(self, response_text: str) -> str:
        """从响应中提取HTML内容"""
        # 尝试找到HTML代码块
        lines = response_text.split('\n')
        html_lines = []
        in_html_block = False
        
        for line in lines:
            # 检测HTML代码块开始
            if '```html' in line.lower() or '```' in line and in_html_block == False and '<!DOCTYPE' in response_text:
                in_html_block = True
                continue
            
            # 检测代码块结束
            if '```' in line and in_html_block:
                break
                
            # 如果在HTML块中，添加行
            if in_html_block:
                html_lines.append(line)
            
            # 如果没有代码块标记但发现HTML开头，直接提取
            if '<!DOCTYPE html>' in line or '<html' in line:
                in_html_block = True
                html_lines.append(line)
        
        # 如果没有找到代码块，尝试直接提取HTML；否则返回空字符串以触发上层重试
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
        
        # 验证HTML完整性
        if not html_content.strip():
            return ""  # 返回空以便上层识别并重试/失败
        
        # 确保有基本的HTML结构
        if '<!DOCTYPE html>' not in html_content and '<html' not in html_content:
            html_content = ""
        
        return html_content
    
     
    def load_app_instruction(self, app_name: str) -> str:
        """加载应用指令"""
        instruction_path = Path(f"examples/{app_name}.yaml")
        with open(instruction_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('prompt', '')
    
    def save_website(self, html_content: str, app_name: str, model_name: str, phase: str = "initial", base_dir: str = "websites") -> str:
        """保存网站文件"""
        if phase == "initial":
            website_dir = Path(f"{base_dir}/{app_name}/{model_name}")
        else:  # revised
            website_dir = Path(f"experiments/{phase}/{app_name}/{model_name}/revised_website")
        
        website_dir.mkdir(parents=True, exist_ok=True)
        
        html_path = website_dir / "index.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(html_path)
