# Computer-Use Agents for Generative UI

<p align="center">
  <a href="https://arxiv.org/abs/2511.15567">📄 Paper</a> &nbsp; | &nbsp;
  <a href="https://huggingface.co/spaces/showlab/AUI">🤗 Demo</a> &nbsp; | &nbsp;
  <a href="https://showlab.github.io/AUI">🌐 Project</a>
</p>

---

## 🔥 Update
- [x] [2025.11.20] Huggingface Demo is released.
- [x] [2025.11.19] Arxiv paper is released.
- [x] [2025.10.30] Code is released.

---

<a id="overview"></a>
## 📖 TL;DR

Can Computer-Use Agents offer feedback to assist Coders to Generate UI?

<div align="center">
  <img src="assets/teaser.png" width="100%" alt=""/>
</div>

---

<a id="quick-start"></a>

## ⚙️ Environments
### 1. Requirements
- Use Python 3.10+ in an isolated environment:
  ```bash
  conda create -n aui python=3.10
  conda activate aui
  ```
- Install dependencies and Playwright browsers:
  ```bash
  pip install -r requirements.txt
  python -m playwright install
  ```

### 2. Configure Models
- Local model servers are recommended (e.g., using [VLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)).
- Edit `configs/models.yaml` to point to your endpoints:
  - **Coder**: `Qwen3-Coder-30B` (`http://localhost:8001/v1`)
  - **CUA**: `UI-TARS-1.5-7B` (`http://localhost:8002/v1`)
  - **Verifier**: `GPT-5` or `Qwen2.5-VL-72B`
- Export API keys (if using proprietary models):
  ```bash
  export AZURE_OPENAI_API_KEY="YOUR_KEY"
  ```

## 🚀 Quick Start

Our AUI contains the following stages.

**Pipeline:**
- **0️⃣ Preparation**: Generate initial websites and tasks per app.
- **1️⃣ Task Solvability Check**: Judge extracts task-state rules on initial websites to determine task validity.
- **2️⃣ CUA Navigation Test**: CUA executes supported tasks; oracle evaluation is rule-based.
- **3️⃣ Iterative Refinement**:
  1. **Revise**: Update websites based on unsupported tasks (Task Solvability Feedback) and CUA failures (Navigation Feedback via Dashboard).
  2. **ReJudge**: Re-evaluate task solvability on revised websites.
  3. **ReTest**: CUA executes tasks on revised websites.
  
For normal usage, you only need the single entrypoint `run.py` from the repo root:

```bash
cd betterui_release
/users/husiyuan/miniconda3/envs/ui/bin/python run.py \
  --models gpt5,qwen,gpt4o \
  --apps all \
  --v0-dir full_52_apps \
  --experiment exp_integrated \
  --revision-type integrated \
  --cua-models uitars
```

This command sequentially runs Stage 0 → Stage 3. Advanced users who want full control over each stage can expand the section below to see what each stage does and the corresponding commands.

<details>
<summary><strong>Show Stage 0–3 details</strong></summary>

**Stage 0 – Preparation**
- `src/stage0_generate_websites.py`: generate initial v0 websites for all apps and coder models.
- `src/stage0_generate_tasks.py`: generate 30 tasks per app (GPT-5) based on app labels.

**Stage 1 – Judge v0 (Task Solvability)**
- `src/stage1_judge_v0.py`: Judge extracts state and completion rules for each task on v0 websites.

**Stage 2 – CUA Test v0 (Navigation)**
- `src/stage2_cua_test_v0.py`: CUA runs only supported tasks (with rules) on v0 websites; success is evaluated by rules (oracle).

**Stage 3 – Revision + Re-eval**
- `src/stage3_0_revise.py`: revise websites using unsupported-task feedback, CUA failures, or integrated signals.
- `src/stage3_1_judge_v1.py`: re-run judge on v1 websites to update task support.
- `src/stage3_2_cua_test_v1.py`: re-run CUA on v1 websites with oracle evaluation.

**1) Generate Initial Websites** (3 coder models × 52 apps)
```bash
python src/stage0_generate_websites.py \
  --models gpt5,qwen,gpt4o \
  --apps all \
  --v0-dir full_52_apps
```

**2) Generate Tasks** (30 tasks per app via GPT-5)
```bash
python src/stage0_generate_tasks.py \
  --apps all \
  --v0-dir full_52_apps
```

**3) Metric 1: Judge Initial Websites** (Task Solvability)
```bash
python src/stage1_judge_v0.py \
  --models gpt5,qwen,gpt4o \
  --apps all \
  --v0-dir full_52_apps
```

**4) Metric 2: CUA Navigation Test** (Initial)
```bash
python src/stage2_cua_test_v0.py \
  --models gpt5,qwen,gpt4o \
  --apps all \
  --v0-dir full_52_apps \
  --cua-models uitars
```

**5) Stage 3: Iterative Refinement** (Choose a revision strategy)

*   **Option A: CUA Revision** (Fix based on navigation failures)
    ```bash
    python src/stage3_0_revise.py \
      --experiment exp_cua_fix \
      --models gpt5,qwen,gpt4o \
      --apps all \
      --revision-type cua \
      --v0-dir full_52_apps
    ```

*   **Option B: Unsupported Task Revision** (Fix based on missing features)
    ```bash
    python src/stage3_0_revise.py \
      --experiment exp_func_fix \
      --models gpt5,qwen,gpt4o \
      --apps all \
      --revision-type unsupported \
      --v0-dir full_52_apps
    ```

*   **Option C: Integrated Revision** (Combine both – Recommended)
    ```bash
    python src/stage3_0_revise.py \
      --experiment exp_integrated \
      --models gpt5,qwen,gpt4o \
      --apps all \
      --revision-type integrated \
      --v0-dir full_52_apps
    ```

**6) Re-evaluate Revised Websites**
```bash
# Re-Judge Task Solvability
python src/stage3_1_judge_v1.py \
  --experiment exp_integrated \
  --models gpt5,qwen,gpt4o \
  --apps all \
  --revision-type integrated \
  --v0-dir full_52_apps

# Re-Run CUA Navigation Test
python src/stage3_2_cua_test_v1.py \
  --experiment exp_integrated \
  --models gpt5,qwen,gpt4o \
  --apps all \
  --revision-type integrated \
  --cua-models uitars \
  --v0-dir full_52_apps
```

</details>

---

<a id="data-layout"></a>
## 🗂️ Data Layout

**Initial Data (Stage 0-2)**
```text
v0/{v0_dir}/
  websites/{app}/{model}/index.html         # Initial Generated Websites
  tasks/{app}/
    tasks.json                              # Generated Tasks
    states/{model}/rules.json               # Stage 1: Validation Rules
    v0_cua_results/{model}/{cua_model}/     # Stage 2: CUA Results
      results.json
      trajectories/task_{i}/step_*.png|json # Trajectories
```

**Experiments (Stage 3)**
```text
experiments/{experiment}/
  runs/{run_key}/
    stage3_0/{app}/{model}/v1_website/index.html    # Revised Websites
    stage3_1/{app}/{model}/rules.json               # Revised Rules
    stage3_2/{cua_model}/{app}/{model}/
      trajectories/{task_id}/                       # New Trajectories
      run_summary.json
```

---

<a id="metrics-components"></a>
## 📏 Evaluations

1.  **Function Completeness (FC)**: Percentage of tasks that are functionally supported by the UI (determined by the Judge).
2.  **CUA Success Rate (SR)**: Percentage of valid tasks successfully completed by the CUA.

**Key Components:**
*   **Verifier**: A GPT-5 based judge that extracts rule-based checks to validate task solvability.
*   **CUA Dashboard**: A visual summary tool that compresses long interaction trajectories into a single image, highlighting key failure points for the Coder.
*   **Revision Strategies**:
    *   `unsupported`: Adds missing features for unsolvable tasks.
    *   `cua`: Fixes usability issues preventing agent navigation (destylization, simplification).
    *   `integrated`: Combines both for maximum performance.

---

<a id="citation"></a>
## 🎓 Citations

If you find this project helpful, please consider citing our paper:

```bibtex
@misc{lin2025aui,
      title={Computer-Use Agents as Judges for Generative User Interface}, 
      author={Kevin Qinghong Lin and Siyuan Hu and Linjie Li and Zhengyuan Yang and Lijuan Wang and Philip Torr and Mike Zheng Shou},
      year={2025},
      eprint={2511.15567},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.15567}, 
}
```

---

<a id="acknowledgements"></a>
## 🙏 Acknowledgements
- Apps are adapted from [OpenAI's coding examples](https://github.com/openai/gpt-5-coding-examples).
- Thanks to the open-source community for browser automation (Playwright) and agent tooling.
