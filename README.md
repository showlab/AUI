# Computer-Use Agents as Judges for Generative User Interface

<!-- <p align="right">
  <b>English</b>
</p> -->

<!-- <p align="center">
  <b>AUI: Computer-Use Agents as Judges for Automatic GUI Design</b><br>
  Apps → Tasks from Agents → CUA Execution → Code Fix by Agents → Revised Apps
</p> -->

<p align="center">
  <a href="#overview">📘 Overview</a> &nbsp; | &nbsp;
  <a href="#quick-start">🚀 Quick Start</a> &nbsp; | &nbsp;
  <a href="#data-layout">🗂️ Data Layout</a> &nbsp; | &nbsp;
  <a href="#metrics-components">📏 Metrics & Components</a>
  <br>
  <a href="https://huggingface.co/spaces/showlab/AUI">🤗 Hugging Face</a> &nbsp; | &nbsp;
  <a href="https://arxiv.org/abs/2511.15567">📄 Paper</a> &nbsp; | &nbsp;
  <a href="https://showlab.github.io/AUI">🌐 Project Website</a> &nbsp; | &nbsp;
</p>

---

## 🔥 Update
- [x] [2025.10.30] Code is released.

---

### Table of Contents
- [📘 Overview](#overview)
- [🚀 Quick Start](#quick-start)
  - [1. Requirements](#1-requirements)
  - [2. Configure Models](#2-configure-models)
  - [3. Run Pipeline (Stage 0 → 3)](#3-run-pipeline-stage-0--3)
- [🗂️ Data Layout](#data-layout)
- [🏗️ Project Organization](#️-project-organization)
- [📏 Metrics & Components](#metrics-components)
- [🧭 Notes & Principles](#-notes--principles)
- [🙏 Acknowledgements](#-acknowledgements)

---

<a id="overview"></a>
## 📘 Overview
AUI is a framework for evaluating agent‑generated web apps end to end.

Pipeline:
- Stage 0 (preparation): generate initial websites (multi‑model, parallel) and 30 tasks per app (GPT‑5).
- Stage 1 (Metric 1): judge extracts task–state rules on initial websites.
- Stage 2 (Metric 2): CUA executes only supported tasks; oracle evaluation is rule‑based (no VLM fallback).
- Stage 3 (Metric 3.1 & 3.2): revise initial websites into revised versions based on failures and unsupported tasks, then re‑judge and re‑CUA.

Principles:
- Parallel by default (#models × #apps), terminal grid status.
- Incremental saving at model–app–task granularity.
- Strictly no fallback branches; retries only for 429 and JSON parsing (≤ 5 attempts each).
- Never truncate HTML or cap task counts.

---

<a id="quick-start"></a>
## 🚀 Quick Start
Run the following commands from this directory (the one containing the stage scripts).

### 1. Requirements
- Use Python 3.10+ in an isolated environment (e.g., `conda create -n ui python=3.10` then `conda activate ui`).
- Local model servers expected by `configs/models.yaml`, using [VLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) to deploy the models is recommended.
  - Qwen2.5‑VL‑72B (commenter ablation): `http://localhost:8000/v1`
  - Qwen3‑Coder‑30B (coder): `http://localhost:8001/v1`
  - UI‑TARS 1.5 7B (CUA): `http://localhost:8002/v1`

Install Python deps and Playwright browsers:
```bash
pip install -r requirements.txt
python -m playwright install
```

### 2. Configure Models
Edit `configs/models.yaml` and export required keys (Azure):
```bash
export AZURE_OPENAI_API_KEY="YOUR_AZURE_KEY"
```
Judge uses GPT‑5 by default.

### 3. Run Pipeline (Stage 0 → 3)
Use `python` from your activated environment.

1) Generate initial websites (3 coder models × 52 apps)
```bash
python stage0_generate_websites.py \
  --models gpt5,qwen,gpt4o \
  --apps all \
  --v0-dir full_52_apps
```

2) Generate 30 tasks per app (GPT‑5)
```bash
python stage0_generate_tasks.py \
  --apps all \
  --v0-dir full_52_apps
```

3) Metric 1 — Judge initial (GPT‑5 as judge)
```bash
python stage1_judge_v0.py \
  --models gpt5,qwen,gpt4o \
  --apps all \
  --v0-dir full_52_apps
```

4) Metric 2 — CUA test initial (oracle eval)
```bash
python stage2_cua_test_v0.py \
  --models gpt5,qwen,gpt4o \
  --apps all \
  --v0-dir full_52_apps \
  --cua-models uitars
```

Stage 3 (repeatable): choose an experiment name and revision type.

5) Stage 3.0 — Revise initial → revised
- CUA revision (destylization + fit‑within‑screen is ON by default)
```bash
python stage3_0_revise.py \
  --experiment exp_new \
  --models gpt5,qwen,gpt4o \
  --apps all \
  --revision-type cua \
  --v0-dir full_52_apps
```
- CUA commenter ablations
```bash
# Text‑only
python stage3_0_revise.py \
  --experiment exp_new \
  --models gpt5,qwen,gpt4o \
  --apps all \
  --revision-type cua \
  --commenter cua-text-only \
  --v0-dir full_52_apps

# Screenshot‑only
python stage3_0_revise.py \
  --experiment exp_new \
  --models gpt5,qwen,gpt4o \
  --apps all \
  --revision-type cua \
  --commenter cua-screenshot-only \
  --v0-dir full_52_apps
```
- Unsupported‑task revision (from Stage 1 judge)
```bash
python stage3_0_revise.py \
  --experiment exp_new \
  --models gpt5,qwen,gpt4o \
  --apps all \
  --revision-type unsupported \
  --v0-dir full_52_apps
```
- Integrated revision (unsupported + CUA)
```bash
python stage3_0_revise.py \
  --experiment exp_new \
  --models gpt5,qwen,gpt4o \
  --apps all \
  --revision-type integrated \
  --v0-dir full_52_apps
```

6) Stage 3.1 — Judge revised
```bash
python stage3_1_judge_v1.py \
  --experiment exp_new \
  --models gpt5,qwen,gpt4o \
  --apps all \
  --revision-type cua \
  --v0-dir full_52_apps
```

7) Stage 3.2 — CUA test revised (per‑task trajectories saved)
```bash
python stage3_2_cua_test_v1.py \
  --experiment exp_new \
  --models gpt5,qwen,gpt4o \
  --apps all \
  --revision-type cua \
  --cua-models uitars \
  --v0-dir full_52_apps
```

---

<a id="data-layout"></a>
## 🗂️ Data Layout
Initial (one‑time prep & eval)
```
v0/{v0_dir}/
  websites/{app}/{model}/index.html
  tasks/{app}/
    tasks.json
    states/{model}/rules.json                 # Stage 1 judge
    v0_cua_results/{model}/{cua_model}/       # Stage 2 CUA
      results.json
      trajectories/task_{i}/step_*.png|json
```

Experiments (repeatable Stage 3 runs)
```
experiments/{experiment}/
  runs/{run_key}/
    stage3_0/{app}/{model}/v1_website/index.html   # revised website
    stage3_1/{app}/{model}/rules.json               # revised rules
    stage3_2/{cua_model}/{app}/{model}/
      trajectories/{task_id}/trajectory.jsonl + step_*.png
      run_summary.json
```
Run keys combine `--revision-type`, `--commenter`, and `--v0-dir` to isolate caches and summaries.
Global and experiment summaries live under `progress/`.

---

<a id="metrics-components"></a>
## 📏 Metrics & Components
- Metric 1 (Stage 1): judge extracts task–state rules on initial websites; score = #tasks with valid rules.
- Metric 2 (Stage 2): CUA executes supported tasks only; oracle eval is rule‑based, no VLM fallback.
- Metric 3 (Stage 3): initial code + failures → storyboard/commenter → coder → revised; re‑run Stage 1 & 2.

Revision Types (Stage 3.0):
- Unsupported: support tasks flagged by Stage 1 judge.
- CUA: revise from CUA failure trajectories. Destylization + fit‑within‑screen enabled by default.
- Integrated: merge Unsupported + CUA.

Commenter Ablations: `cua-text-only`, `cua-screenshot-only`.

Storyboard: 1920×1080 canvas with dynamic crops; saved next to each trajectory.

CUA termination: early stop after 3 consecutive steps with no state change; per‑task trajectories and JSONL are saved.

---

## 🏗️ Project Organization
```
agents/                 # coder, judge, CUA policies, commenters
revision_components/    # unsupported, CUA-based, integrated revisions
utils/                  # model client, parallel runner, progress tracker, etc.
configs/models.yaml     # model endpoints and keys
stage0_generate_websites.py
stage0_generate_tasks.py
stage1_judge_v0.py
stage2_cua_test_v0.py
stage3_0_revise.py
stage3_1_judge_v1.py
stage3_2_cua_test_v1.py
v0/                     # generated websites, tasks, states, and CUA results for initial websites
experiments/            # stage 3 runs (revised websites, rules, and CUA results)
progress/               # global/experiment summaries and evaluations
```

---

## 🧭 Notes & Principles
- Judge currently uses GPT‑5.

## 🙏 Acknowledgements
- Apps are adapted from this repository’s examples; websites are generated by coder models.
- Thanks to the open‑source community for browser automation and agent tooling.
 
