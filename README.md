# Sean McVAI: Beating Two-High Safety Looks with AI-Designed Play Calling

> **Football Analytics Blitz 2026** — An LLM feedback loop trained on real NFL play-by-play data, an LSTM predictability model, and a LightGBM expected-yards model to mathematically optimize pass plays against two-high safety coverages.

---

## Overview

The Football Analytics Blitz 2026 prompt asked us to analyze defensive two-high safety schemes from both an offensive and defensive perspective, and to design a pass play to beat them. Our group — **Jack Kalsched, Leonard Malott, Pranav Rajaram, Diego Osborn, and Adrian Rodriguez** — took the prompt a step further.

We built **Sean McVAI**: a closed-loop AI system that *generates, scores, and iteratively refines* pass plays optimized against two-high looks in any game situation. The system is grounded in three data-driven components:

1. **2025 Play-by-Play Dataset** — provided by Sports Info Solutions, used for model training and similarity search
2. **LSTM Recurrent Neural Network** — produces a *scheme predictability score* for each proposed play, penalizing tendencies the defense can read
3. **LightGBM Model** — produces a continuous *expected yards gained* estimate for each play

Through reverse-prompting and repeated evaluation, the LLM designs plays that are mathematically optimal against two-high safety looks for any down, distance, and field position.

---

## System Architecture

The core of Sean McVAI is a **generator → critic → rationale → memory** feedback loop:

```
Game Context
     │
     ▼
┌─────────────┐       ┌──────────────────────────────┐
│  Generator  │──────▶│           Critic              │
│   (LLM)     │       │  LSTM Predictability Score    │
│             │◀──────│  LightGBM Expected Yards      │
└─────────────┘       └──────────────────────────────┘
      │  ▲
      │  │  Self-reflection + memory
      ▼  │
┌─────────────┐
│  Rationale  │  "Why did I score this way? What should I change?"
│   (LLM)     │
└─────────────┘
      │
      ▼
 Best play selected by: E[Yards] − λ × Predictability
```

### Components

| Component | Path | Description |
|---|---|---|
| **Orchestrator** | `v2/orchestrator.py` | Runs the full generate → critique → rationale loop for N iterations, saves trace to `v2_feedback_trace.json` |
| **Play Generator** | `v2/play_generator.py` | LLM that proposes plays (formation, personnel, route concepts) as structured JSON, with memory of past attempts |
| **Critic** | `v2/critic.py` | Scores each proposed play via the LSTM (predictability) and LightGBM (expected yards), returns textual critique |
| **Streamlit Dashboard** | `streamlit/demo_streamlit.py` | Interactive "Sean McVAI: Beating 2-High" app combining all models in one UI |
| **Web Visualization** | `web_viz/` | React + Vite front-end for formation selection, route concepts, and two-high attack metrics |
| **Modeling Notebooks** | `modeling/`, `eda/` | EDA, LSTM predictability training, LightGBM expected-yards training |

---

## Quickstart

### Prerequisites

- Python 3.9+ with PyTorch, LightGBM, Streamlit, and standard data science dependencies
- Node.js + npm (for the web visualization)
- An `OPENAI_API_KEY` configured in your environment (for the LLM feedback loop)

---

### 1. LLM Feedback Loop (Sean McVAI Core)

Ensure model files (`predictability_lstm.pkl`, `predictability_lstm.pt`, `quantile_model/`) are present in the project root, then:

```bash
cd football-analytics-blitz-2026
python -m v2.orchestrator
```

Or from inside `v2/`:

```bash
cd football-analytics-blitz-2026/v2
python orchestrator.py
```

After the loop completes, inspect `v2_feedback_trace.json` for the full iteration trace — each proposed play, its critic scores, critique, and the LLM's rationale — and see the selected best play printed to the console.

> **Tip:** Adjust the number of iterations, the composite scoring weight λ, and model paths in `v2/config.py`.

---

### 2. Streamlit Dashboard

```bash
cd football-analytics-blitz-2026
streamlit run streamlit/demo_streamlit.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`). The dashboard lets you interactively explore predictability scores, expected yards, and optimal two-high attack strategies across game scenarios.

> **Tip:** Run from the project root so that `quantile_model/` and `predictability_lstm.pkl` resolve correctly. If you see missing-file errors, check paths in `demo_streamlit.py`.

---

### 3. Web Visualization

```bash
cd football-analytics-blitz-2026/web_viz
npm install
npm run dev
```

Open the URL Vite prints (typically `http://localhost:5173`) for interactive formation and route visualizations.

**Production build:**
```bash
npm run build
npm run preview   # optional: preview locally before deploy
```

---

## Models

### LSTM Predictability Model
- **Architecture:** Recurrent Neural Network (LSTM) trained on sequential play-by-play data
- **Output:** A continuous predictability score — lower scores indicate plays that are harder for the defense to anticipate
- **Artifact:** `predictability_lstm.pt` (weights), `predictability_lstm.pkl` (serialized model)

### LightGBM Expected Yards Model
- **Architecture:** Gradient-boosted decision tree (LightGBM) trained on down, distance, field position, personnel, and formation features
- **Output:** Continuous expected yards gained for each play design
- **Artifact:** `quantile_model/`

### Composite Scoring
The best play is selected by:

```
Score = E[Yards] − λ × Predictability
```

where λ balances unpredictability against raw yardage expectation — keeping the offense both effective and hard to read.

---

## Repository Structure

```
football-analytics-blitz-2026/
├── v2/                          # LLM feedback loop (core system)
│   ├── orchestrator.py          # Main loop entrypoint
│   ├── play_generator.py        # LLM play proposal + self-reflection
│   ├── critic.py                # LSTM + LightGBM scoring
│   ├── critic_model_def.py      # Model definitions
│   ├── config.py                # Paths, model names, hyperparameters
│   └── system_prompt.txt        # LLM system prompt
├── streamlit/
│   └── demo_streamlit.py        # Interactive Streamlit dashboard
├── web_viz/                     # React + Vite front-end
├── modeling/                    # Expected yards training notebooks
├── eda/                         # Exploratory data analysis
├── quantile_model/              # Serialized LightGBM model artifacts
├── predictability_lstm.pt       # LSTM weights
├── predictability_lstm.pkl      # LSTM serialized model
└── v2_feedback_trace.json       # Output: full LLM iteration trace
```

---

## Data

Play-by-play data for the 2025 NFL season was provided by **Sports Info Solutions** as part of the Football Analytics Blitz 2026 competition. The dataset is not included in this repository.

---

*Built for the 2026 Football Analytics Blitz.*
