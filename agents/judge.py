import json
from typing import Dict, Any, List
from pathlib import Path
from .prompts.judge_prompts import (
    build_analyze_prompt,
    build_analyze_three_component_prompt,
    build_single_rule_prompt,
)

class Judge:
    """Judge Agent - 使用GPT-5提取状态和生成规则"""
    
    def __init__(self, model_client=None):
        self.model_client = model_client
    
    async def analyze_website_tasks(self, app_name: str, html_content: str, 
                             tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析网站和任务，提取状态规则"""
        
        try:
            # 构建任务列表
            tasks_text = "\n".join([
                f"{i+1}. {task.get('description', '')}"
                for i, task in enumerate(tasks)
            ])
            
            prompt = build_analyze_prompt(html_content, tasks_text)

            # JSON解析与结构校验重试机制
            def _valid_schema(obj) -> bool:
                if not isinstance(obj, list):
                    return False
                required = ['task_index', 'task_description', 'supportable', 'rule']
                for item in obj:
                    if not isinstance(item, dict):
                        return False
                    for k in required:
                        if k not in item:
                            return False
                return True

            task_rules = None
            for attempt in range(5):
                try:
                    if attempt > 0:
                        # 重试时强调JSON格式与字段要求
                        prompt += f"\n\nIMPORTANT: Output valid JSON array only with objects containing keys: task_index, task_description, supportable, rule, expected_outcome, reason. Attempt {attempt + 1}/5."
                    response = await self.model_client.call_judge(prompt)
                    content = response
                    # 提取JSON部分
                    if '```json' in content:
                        content = content.split('```json')[1].split('```')[0]
                    elif '```' in content:
                        content = content.split('```')[1].split('```')[0]
                    task_rules = json.loads(content)
                    # 结构校验：若不合规，视为解析失败进入重试
                    if not _valid_schema(task_rules):
                        if attempt == 4:
                            return {
                                'success': False,
                                'error': "Invalid JSON schema after 5 attempts (expected list of objects with required keys)"
                            }
                        continue
                    break
                except (json.JSONDecodeError, IndexError) as e:
                    if attempt == 4:  # 最后一次尝试失败
                        return {
                            'success': False,
                            'error': f"Failed to parse JSON after 5 attempts: {str(e)}"
                        }
                    continue
            
            # 统计
            supported = [t for t in task_rules if t.get('supportable')]
            unsupported = [t for t in task_rules if not t.get('supportable')]
            
            return {
                'success': True,
                'analysis': {
                    'supported_tasks': supported,
                    'unsupported_tasks': unsupported,
                },
                'supported_count': len(supported),
                'unsupported_count': len(unsupported),
                'total_tasks': len(tasks)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def analyze_website_tasks_three_component(self, app_name: str, html_content: str, 
                             tasks: List[Dict[str, Any]], component: str = "full") -> Dict[str, Any]:
        """三组件分析 - 支持task_description, expected_outcome, reason的不同组合"""
        
        try:
            # 构建任务列表
            tasks_text = "\n".join([
                f"{i+1}. {task.get('description', '')}"
                for i, task in enumerate(tasks)
            ])
            
            # 根据组件类型调整prompt
            if component == "description_only":
                analysis_instruction = """Analyze each task based ONLY on the task description. Do not infer expected outcomes or provide detailed reasoning.

Analyze each task and respond with a JSON array. Each item must have exactly these fields:
- task_index: number (1-based)
- task_description: string (copy the original task description)
- supportable: true/false
- rule: simple completion rule if supportable (empty string if not supportable)

Do not include expected_outcome or reason fields."""
                
            elif component == "description_outcome":
                analysis_instruction = """Analyze each task based on the task description and infer the expected outcome. Do not provide detailed reasoning.

Analyze each task and respond with a JSON array. Each item must have exactly these fields:
- task_index: number (1-based)
- task_description: string (copy the original task description)
- expected_outcome: string (describe what successful completion should look like)
- supportable: true/false
- rule: simple completion rule if supportable (empty string if not supportable)

Do not include reason field."""
                
            else:  # full
                analysis_instruction = """Analyze each task thoroughly and respond with a JSON array. Each item must have exactly these fields:
- task_index: number (1-based)
- task_description: string (copy the original task description)
- expected_outcome: string (describe what successful completion should look like)
- supportable: true/false
- rule: simple completion rule if supportable (empty string if not supportable)
- reason: detailed explanation of why the task is or isn't supportable (minimum 2-3 sentences)

Include all six fields for each task."""
            
            prompt = build_analyze_three_component_prompt(html_content, tasks_text, analysis_instruction)

            # JSON解析重试机制
            task_rules = None
            for attempt in range(5):
                try:
                    if attempt > 0:
                        # 重试时强调JSON格式
                        prompt += f"\n\nIMPORTANT: You must output valid JSON only. This is attempt {attempt + 1}/5."
                    
                    response = await self.model_client.call_judge(prompt)
                    
                    # 解析响应
                    content = response
                    # 提取JSON部分
                    if '```json' in content:
                        content = content.split('```json')[1].split('```')[0]
                    elif '```' in content:
                        content = content.split('```')[1].split('```')[0]
                    
                    task_rules = json.loads(content)
                    break
                except (json.JSONDecodeError, IndexError) as e:
                    if attempt == 4:  # 最后一次尝试失败
                        return {
                            'success': False,
                            'error': f"Failed to parse JSON after 5 attempts: {str(e)}"
                        }
                    continue
            
            # 统计
            supported = [t for t in task_rules if t.get('supportable')]
            unsupported = [t for t in task_rules if not t.get('supportable')]
            
            return {
                'success': True,
                'component': component,
                'analysis': {
                    'supported_tasks': supported,
                    'unsupported_tasks': unsupported,
                },
                'supported_count': len(supported),
                'unsupported_count': len(unsupported),
                'total_tasks': len(tasks)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def generate_task_completion_rule(self, task_description: str, 
                                     html_content: str) -> Dict[str, Any]:
        """为单个任务生成完成规则"""
        
        prompt = build_single_rule_prompt(task_description, html_content)

        # JSON解析重试机制
        for attempt in range(5):
            if attempt > 0:
                prompt += f"\n\nIMPORTANT: You must output valid JSON only. This is attempt {attempt + 1}/5."
            
            response = await self.model_client.call_judge(prompt)
            
            content = response
            if '```' in content:
                content = content.split('```')[1].split('```')[0]
            
            rule = json.loads(content)
            return rule
    
    def evaluate_task_completion(self, rule_str: str, page_state: Dict[str, Any]) -> bool:
        """评估任务是否完成（增强版: 支持属性选择器与更严格的exists语义）"""
        if not rule_str:
            return False
        # 处理复合规则（AND/OR逻辑）
        if ' AND ' in rule_str:
            conditions = rule_str.split(' AND ')
            return all(self._evaluate_single_condition(cond.strip(), page_state) for cond in conditions)
        elif ' OR ' in rule_str:
            conditions = rule_str.split(' OR ')
            return any(self._evaluate_single_condition(cond.strip(), page_state) for cond in conditions)
        else:
            return self._evaluate_single_condition(rule_str, page_state)
    
    def _evaluate_single_condition(self, condition: str, page_state: Dict[str, Any]) -> bool:
        """评估单个条件（支持 #id[attr] / #id[attr^='x'] / #id exists 等）"""
        if not condition:
            return False
        condition = condition.strip()

        # ---- Extended helpers (state-only) ----
        SUFFIX_KEYS = ("_visible", "_class", "_data", "_aria", "_attr")

        def _all_ids():
            ids = set()
            for k in page_state.keys():
                if k.startswith("__meta_"):
                    continue
                matched_suffix = False
                for suf in SUFFIX_KEYS:
                    if k.endswith(suf):
                        ids.add(k[: -len(suf)])
                        matched_suffix = True
                        break
                if not matched_suffix:
                    ids.add(k)
            return ids

        def _class_exists(class_name: str) -> bool:
            cls = class_name.strip().lstrip('.')
            if not cls:
                return False
            for k, v in page_state.items():
                if k.endswith("_class") and isinstance(v, str) and cls in v:
                    return True
            return False

        def _attr_matches(op: str, actual: str, expected: str) -> bool:
            actual = actual or ''
            if op == '^=':
                return actual.startswith(expected)
            if op == '$=':
                return actual.endswith(expected)
            if op == '*=':
                return expected in actual
            if op in ('=', '=='):
                return actual == expected
            if op == '!=':
                return actual != expected
            return False

        def _scan_global_attr(attr_name: str, op: str = None, val: str = None) -> bool:
            for k, v in page_state.items():
                if not k.endswith("_attr") or not isinstance(v, dict):
                    continue
                av = str((v or {}).get(attr_name, ''))
                if op is None:
                    if av != '':
                        return True
                else:
                    if _attr_matches(op, av, val):
                        return True
            return False

        def _get_text(el_id: str) -> str:
            return str(page_state.get(el_id, ''))

        # Helper: element presence independent of text content
        def _id_exists(el_id: str) -> bool:
            if not el_id:
                return False
            return (
                (el_id in page_state) or
                (f"{el_id}_visible" in page_state) or
                (f"{el_id}_class" in page_state) or
                (f"{el_id}_aria" in page_state) or
                (f"{el_id}_data" in page_state) or
                (f"{el_id}_attr" in page_state)
            )

        # Helper: parse "#id[... ]" into components
        def _parse_id_attr(expr: str):
            if '[' in expr and ']' in expr:
                before, after = expr.split('[', 1)
                el_id = before.strip().lstrip('#')
                inside = after.split(']')[0].strip()
                name = inside
                op = None
                val = None
                for candidate in ("^=", "$=", "*=", "==", "!=", "="):
                    if candidate in inside:
                        parts = inside.split(candidate, 1)
                        name = parts[0].strip()
                        op = candidate
                        val = parts[1].strip().strip("\"'")
                        break
                return el_id, name, op, val
            return expr.strip().lstrip('#'), None, None, None

        # Visibility sugar: only match exact forms "#id visible" or "#id not visible"
        # Guard against text conditions like "#status text contains visible"
        import re
        m_vis = re.match(r"^\s*#([A-Za-z_][\w\-]*)\s+visible\s*$", condition)
        if m_vis:
            el_id = m_vis.group(1)
            vis = bool(page_state.get(f"{el_id}_visible", False))
            return vis
        m_not_vis = re.match(r"^\s*#([A-Za-z_][\w\-]*)\s+not\s+visible\s*$", condition)
        if m_not_vis:
            el_id = m_not_vis.group(1)
            vis = bool(page_state.get(f"{el_id}_visible", False))
            return (not vis)

        # Equality/inequality with attribute selector: "#id[aria-disabled] == 'true'"
        if ' == ' in condition or ' != ' in condition:
            op = ' == ' if ' == ' in condition else ' != '
            left, right = condition.split(op, 1)
            left = left.strip()
            expected = right.strip().strip("\"'")
            el_id, attr_name, attr_op, attr_val = _parse_id_attr(left)
            if not el_id:
                return False
            if attr_name:
                # aria-* uses _aria map; others from _attr map
                if attr_name.startswith('aria-'):
                    aria = page_state.get(f"{el_id}_aria", {}) or {}
                    actual = str(aria.get(attr_name[5:], ''))
                else:
                    attrs = page_state.get(f"{el_id}_attr", {}) or {}
                    actual = str(attrs.get(attr_name, ''))
                return (actual == expected) if op.strip() == '==' else (actual != expected)
            else:
                actual = str(page_state.get(el_id, ''))
                return (actual == expected) if op.strip() == '==' else (actual != expected)

        # 处理复杂条件如 "#color-word text != ''"，以及 contains/startswith/endswith 变体
        if ' text ' in condition:
            # 提取元素ID和操作
            if ' text !=' in condition:
                parts = condition.split(' text !=')
                element_id = parts[0].strip('#')
                expected = parts[1].strip().strip("'\"")
                return _get_text(element_id) != expected
            if ' text ==' in condition:
                parts = condition.split(' text ==')
                element_id = parts[0].strip('#')
                expected = parts[1].strip().strip("'\"")
                return _get_text(element_id) == expected
            if ' text contains ' in condition:
                parts = condition.split(' text contains ')
                element_id = parts[0].strip('#')
                expected = parts[1].strip().strip("'\"")
                return expected in _get_text(element_id)
            if ' text icontains ' in condition:
                parts = condition.split(' text icontains ')
                element_id = parts[0].strip('#')
                expected = parts[1].strip().strip("'\"")
                return expected.lower() in _get_text(element_id).lower()
            if ' text startswith ' in condition:
                parts = condition.split(' text startswith ')
                element_id = parts[0].strip('#')
                expected = parts[1].strip().strip("'\"")
                return _get_text(element_id).startswith(expected)
            if ' text endswith ' in condition:
                parts = condition.split(' text endswith ')
                element_id = parts[0].strip('#')
                expected = parts[1].strip().strip("'\"")
                return _get_text(element_id).endswith(expected)

        # Attribute presence/prefix/suffix/substring with exists: "#id[attr^='x'] exists"
        if condition.endswith(' exists'):
            left = condition[:-6].strip()
            # .class / #id .class
            if left.startswith('.') or (' .' in left):
                cls = left.split('.')[-1]
                return _class_exists(cls)
            # [attr...] / #id [attr...] → 全局属性扫描
            if left.startswith('[') or ('[' in left and ']' in left):
                inside = left[left.find('[')+1 : left.rfind(']')].strip()
                name = inside
                op = None
                val = None
                for candidate in ("^=", "$=", "*=", "==", "!=", "="):
                    if candidate in inside:
                        parts = inside.split(candidate, 1)
                        name = parts[0].strip()
                        op = candidate
                        val = parts[1].strip().strip("\"'")
                        break
                if name == 'id' and op in ('^=',):
                    pref = val or ''
                    return any(i.startswith(pref) for i in _all_ids())
                return _scan_global_attr(name, op, val)
            # 默认：#id[attr...] 或 #id exists
            el_id, attr_name, attr_op, attr_val = _parse_id_attr(left)
            if not el_id:
                return False
            if attr_name:
                if attr_name.startswith('aria-'):
                    aria = page_state.get(f"{el_id}_aria", {}) or {}
                    v = str(aria.get(attr_name[5:], ''))
                else:
                    attrs = page_state.get(f"{el_id}_attr", {}) or {}
                    v = str(attrs.get(attr_name, ''))
                if attr_op is None:
                    return v != ''
                return _attr_matches(attr_op, v, attr_val)
            return _id_exists(el_id)
        
        # 样式相关/点击文案等无法从纯state评估的条件不再自动通过
        if 'getComputedStyle' in condition or 'background-color' in condition or 'Clicked' in condition:
            return False
        
        # CSS规则检查不做自动通过
        if 'Stylesheet contains' in condition:
            return False
        
        # 处理简单规则（扩展：icontains/startswith/endswith/比较符）
        if ' icontains ' in condition:
            parts = condition.split(' icontains ')
            if len(parts) == 2:
                element_id = parts[0].strip('#')
                expected = parts[1].strip("'\"")
                actual = str(page_state.get(element_id, ''))
                return expected.lower() in actual.lower()
        if ' startswith ' in condition:
            parts = condition.split(' startswith ')
            if len(parts) == 2:
                element_id = parts[0].strip('#')
                expected = parts[1].strip("'\"")
                actual = str(page_state.get(element_id, ''))
                return actual.startswith(expected)
        if ' endswith ' in condition:
            parts = condition.split(' endswith ')
            if len(parts) == 2:
                element_id = parts[0].strip('#')
                expected = parts[1].strip("'\"")
                actual = str(page_state.get(element_id, ''))
                return actual.endswith(expected)
        if 'contains' in condition:
            parts = condition.split(' contains ')
            if len(parts) == 2:
                element_id = parts[0].strip('#')
                expected = parts[1].strip("'\"")
                actual = str(page_state.get(element_id, ''))
                return expected in actual
        # numeric comparisons
        for op in (' >= ', ' <= ', ' < ', ' > '):
            if op in condition:
                left, right = condition.split(op, 1)
                element_id = left.strip('#')
                try:
                    expected = float(right.strip())
                except Exception:
                    return False
                try:
                    actual = float(page_state.get(element_id, 0))
                except Exception:
                    # 从文本中尝试提取首个数字
                    import re
                    m = re.search(r"-?\d+(?:\.\d+)?", str(page_state.get(element_id, '')))
                    actual = float(m.group(0)) if m else 0.0
                if op.strip() == '>':
                    return actual > expected
                if op.strip() == '<':
                    return actual < expected
                if op.strip() == '>=':
                    return actual >= expected
                if op.strip() == '<=':
                    return actual <= expected
                return False
        
        if ' == ' in condition:
            parts = condition.split(' == ')
            if len(parts) == 2:
                element_id = parts[0].strip('#')
                expected = parts[1].strip("'\"")
                actual = str(page_state.get(element_id, ''))
                return actual == expected
        
        if ' != ' in condition:
            parts = condition.split(' != ')
            if len(parts) == 2:
                element_id = parts[0].strip('#')
                expected = parts[1].strip("'\"")
                actual = str(page_state.get(element_id, ''))
                return actual != expected
        
        if 'exists' in condition:
            element_id = condition.split(' exists')[0].strip('#')
            return _id_exists(element_id)
        
        return False
    
    def save_rules(self, app_name: str, model_name: str, rules: Dict[str, Any], version: str = "initial", v0_dir: str = None):
        """保存规则到文件"""
        if version == "initial":
            if v0_dir:
                rules_dir = Path(f"initial/{v0_dir}/tasks/{app_name}/states/{model_name}")
            else:
                rules_dir = Path(f"tasks/{app_name}/states/{model_name}")
        else:
            rules_dir = Path(f"experiments/{version}/{app_name}/{model_name}")
        
        rules_dir.mkdir(parents=True, exist_ok=True)
        
        rules_file = rules_dir / ("rules.json" if version == "initial" else "revised_rules.json")
        with open(rules_file, 'w', encoding='utf-8') as f:
            json.dump(rules, f, indent=2, ensure_ascii=False)
        
        return str(rules_file)
