import json
import re
from pathlib import Path
from typing import Dict, Any, List


ID_REGEX = re.compile(r"#[A-Za-z_][A-Za-z0-9_\-]*")
CONTAINS_REGEX_1 = re.compile(r"#([A-Za-z_][A-Za-z0-9_\-]*)\s+contains\s+'([^']+)'", re.IGNORECASE | re.MULTILINE)
CONTAINS_REGEX_2 = re.compile(r"#([A-Za-z_][A-Za-z0-9_\-]*)\s+contains\s+\"([^\"]+)\"", re.IGNORECASE | re.MULTILINE)
FUNC_DEF_REGEX = re.compile(r"function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
ADD_LISTENER_REGEX = re.compile(r"addEventListener\s*\(\s*['\"]([A-Za-z]+)['\"]")
INLINE_ON_REGEX = re.compile(r"\son([a-z]+)=\"")


def _unique(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_contract(v0_rules_path: Path, v0_html: str) -> Dict[str, Any]:
    """Extract a minimal non-regression contract from Stage 1 rules and initial HTML.

    - keep_selectors: list of #ids referenced by supported rules
    - keep_text_contains: list of {selector, text} from `contains` rules
    - keep_api: function names and generic event tokens observed in initial (static scan)
    - notes: generic constraints about initial state / viewport
    """
    keep_selectors: List[str] = []
    keep_text_contains: List[Dict[str, str]] = []
    keep_api: List[Dict[str, str]] = []
    notes: List[str] = [
        "no_auto_trigger_on_load",
        "fit_controls_in_1280x720",
        "keep_initial_neutral_state",
        "do_not_tighten_validation",
    ]

    rules_data = None
    if v0_rules_path and v0_rules_path.exists():
        rules_data = json.loads(v0_rules_path.read_text(encoding="utf-8"))
        supported = rules_data.get("analysis", {}).get("supported_tasks", [])
        for item in supported:
            rule = item.get("rule", "") or ""
            if not rule:
                continue
            # Collect #ids
            keep_selectors += ID_REGEX.findall(rule)
            # Collect contains text
            for m in CONTAINS_REGEX_1.finditer(rule):
                sel = m.group(1)
                txt = m.group(2)
                if sel and txt:
                    keep_text_contains.append({"selector": f"#{sel}", "text": txt})
            for m in CONTAINS_REGEX_2.finditer(rule):
                sel = m.group(1)
                txt = m.group(2)
                if sel and txt:
                    keep_text_contains.append({"selector": f"#{sel}", "text": txt})

    keep_selectors = _unique(keep_selectors)

    # Initial API/function/event tokens
    if v0_html:
        funcs = _unique(FUNC_DEF_REGEX.findall(v0_html))
        for fn in funcs:
            keep_api.append({"type": "function", "name": fn})
        # We keep event names as generic hints (not element-bound)
        evts = _unique(ADD_LISTENER_REGEX.findall(v0_html))
        for ev in evts:
            keep_api.append({"type": "event", "name": ev})
        # Inline events presence
        inline = _unique(INLINE_ON_REGEX.findall(v0_html))
        for ev in inline:
            keep_api.append({"type": "inline_event", "name": ev})

    return {
        "keep_selectors": keep_selectors,
        "keep_text_contains": keep_text_contains,
        "keep_api": keep_api,
        "notes": notes,
    }


def format_contract_prompt(contract: Dict[str, Any]) -> str:
    """Format contract into a concise prompt section."""
    return (
        "\n\n## CODE PRESERVATION CONTRACT (Non-Regression)\n"
        + json.dumps(contract, ensure_ascii=False, indent=2)
        + "\n\nRules:\n"
        "- Do not rename/remove selectors in keep_selectors; keep their semantics.\n"
        "- Keep texts in keep_text_contains emitted by the same selectors on success.\n"
        "- Do not introduce stricter validation that blocks task inputs.\n"
        "- Do not auto-trigger flows on load; keep initial neutral state.\n"
        "- Fit critical controls within 1280x720 without scrolling.\n"
        "- Add new features via new IDs; do not overwrite existing ones.\n"
    )


def validate_revised(v1_html: str, contract: Dict[str, Any]) -> Dict[str, Any]:
    """Static validation of revised HTML against the minimal contract.

    Checks:
      - missing_selectors: any required #id absent
      - missing_api: required function names absent
    """
    missing_selectors: List[str] = []
    missing_api: List[str] = []

    if not v1_html:
        return {
            "valid": False,
            "missing_selectors": contract.get("keep_selectors", []) or [],
            "missing_api": [a.get("name") for a in contract.get("keep_api", []) if a.get("type") == "function"],
        }

    # Check id existence (simple static check)
    for sel in contract.get("keep_selectors", []) or []:
        # Convert #id to pattern id="id" or id='id'
        if not re.search(rf'id\s*=\s*["\\\']{re.escape(sel[1:])}["\\\']', v1_html):
            missing_selectors.append(sel)

    # Check function names presence
    for api in contract.get("keep_api", []) or []:
        if api.get("type") == "function":
            name = api.get("name")
            if name and not re.search(rf"function\s+{re.escape(name)}\s*\(", v1_html):
                missing_api.append(name)

    valid = not missing_selectors and not missing_api
    return {"valid": valid, "missing_selectors": missing_selectors, "missing_api": missing_api}


def save_json(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
