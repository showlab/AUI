"""
Stage 3.0 revise runner helpers extracted from stage3_0_revise.py to reduce file size.
Keep behavior and interfaces identical.
"""

from typing import Dict, Any
from pathlib import Path
import json


def save_revised_website(html_content: str, app_name: str, model_name: str,
                         experiment_name: str, run_key: str, meta: Dict[str, Any]) -> str:
    """Save revised website under runs/[run_key]/stage3_0 (uses revised_website path)."""
    website_dir = Path(f"experiments/{experiment_name}/runs/{run_key}/stage3_0/{app_name}/{model_name}/revised_website")
    website_dir.mkdir(parents=True, exist_ok=True)
    website_path = website_dir / 'index.html'
    with open(website_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    if meta:
        meta_path = website_dir / 'meta.json'
        with open(meta_path, 'w', encoding='utf-8') as mf:
            json.dump(meta, mf, indent=2, ensure_ascii=False)
    return str(website_path)


def build_variant_name(revision_type: str, destylized: bool) -> str:
    if revision_type in ['cua', 'integrated']:
        suffix = 'destylized'
    else:
        suffix = 'plain'
    return f"{revision_type}_{suffix}"
