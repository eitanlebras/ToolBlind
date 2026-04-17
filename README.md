# ToolBlind: Measuring AI Agent Reasoning Under Tool Absence with Trajectory Commitment

ToolBlind is a **REST API and benchmark** for evaluating whether AI agents can reason correctly when a required tool becomes unavailable mid-trajectory. It exposes a programmable interface for browsing 500 benchmark tasks, running them against stub or live agents, and computing evaluation metrics — all via HTTP endpoints.

## API

**Base URL:** Deploy and call directly. No API keys required for the test API.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API overview and version |
| `GET` | `/stats` | Dataset statistics (500 tasks, 5 domains, 3 tiers) |
| `GET` | `/tasks` | List/filter tasks by tier, domain, outcome |
| `GET` | `/tasks/{task_id}` | Full task detail with tools and steps |
| `GET` | `/run/{task_id}` | Run a single task with a stub agent |
| `POST` | `/run/batch` | Run N tasks and get aggregate metrics |

**Interactive docs:** `GET /docs` (Swagger UI) or `GET /redoc` (ReDoc)

**OpenAPI spec:** [`openapi.json`](openapi.json) included in the repo.

### Run locally

```bash
pip install -e .
uvicorn api:app --reload
# Open http://localhost:8000/docs
```

### Example requests

```bash
# Get dataset stats
curl http://localhost:8000/stats

# List Tier 1 web tasks
curl "http://localhost:8000/tasks?tier=1&domain=web&limit=5"

# Run a task with the "smart" strategy
curl http://localhost:8000/run/tb_t1_web_0000?strategy=smart

# Batch run 50 tasks and get ToolBlind Score
curl -X POST "http://localhost:8000/run/batch?sample=50&strategy=smart"
```

---

## Benchmark Overview

ToolBlind tests whether agents can recognize tool *absence* and respond appropriately — by substituting, decomposing, or halting — rather than confabulating a path forward. The key independent variable is *trajectory commitment*: how many steps has the agent already completed before encountering the gap?

## Installation (full benchmark)

```bash
git clone https://github.com/eitanlebras/ToolBlind.git
cd toolblind
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your API keys
```

## Quick Start

Generate the dataset and run a quick baseline:

```bash
# Generate the 500-task dataset
python scripts/generate_dataset.py

# Run baseline experiment on 10 tasks per tier (fast development mode)
python scripts/run_experiment.py baseline --models claude --sample 10

# Analyze results
python scripts/analyze_results.py --latest
```

## Full Experiment Reproduction

```bash
# Generate dataset (required first)
python scripts/generate_dataset.py --seed 42

# Run all 5 experiments with all models
python scripts/run_all_experiments.py --models claude openai gemini

# Or run individually:
python scripts/run_experiment.py baseline --models claude openai gemini
python scripts/run_experiment.py commitment --models claude openai gemini
python scripts/run_experiment.py framing --models claude openai gemini
python scripts/run_experiment.py registry_size --models claude openai gemini
python scripts/run_experiment.py cot --models claude openai gemini

# Analyze any results file
python scripts/analyze_results.py data/results/baseline_*.json
```

## Dataset Statistics

| Category | Count |
|----------|-------|
| Total tasks | 500 |
| Tier 1 (Substitution) | 175 |
| Tier 2 (Decomposition) | 175 |
| Tier 3 (Infeasibility) | 150 |
| Domains | 5 (web, code, file, api, database) |
| Tasks per domain | 100 |
| Ablation subset | 100 tasks x 4 framings x 4 registry sizes |
| Commitment variants | 150 tasks x 5 depths |

## Metrics

| Metric | Description |
|--------|-------------|
| TBS (ToolBlind Score) | Weighted accuracy across tiers (w1=0.25, w2=0.35, w3=0.40) |
| CR (Confabulation Rate) | Fraction of tasks where agent confabulates |
| FSR (False Substitution Rate) | Fraction using non-equivalent substitutes |
| FRS (Functional Reasoning Score) | Quality of reasoning (0-3 scale) |
| ECE (Expected Calibration Error) | Confidence-accuracy calibration |

## Results

*(To be populated after experimental runs)*

| Model | TBS | CR | CR-T1 | CR-T2 | CR-T3 | FSR | FRS |
|-------|-----|-----|-------|-------|-------|-----|-----|
| Claude Sonnet | — | — | — | — | — | — | — |
| GPT-4o | — | — | — | — | — | — | — |
| Gemini 1.5 Pro | — | — | — | — | — | — | — |

## Experiments

1. **Baseline**: Full evaluation across all tiers and models
2. **Commitment Depth**: Effect of prior trajectory commitment on confabulation
3. **Framing Ablation**: How unavailability reason framing affects agent behavior
4. **Registry Size**: Effect of available tool count on confabulation
5. **CoT vs Direct**: Chain-of-thought impact on tool absence reasoning

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=toolblind --cov-report=term-missing
```

## Citation

```bibtex
@inproceedings{toolblind2025,
  title={ToolBlind: Measuring AI Agent Reasoning Under Tool Absence with Trajectory Commitment},
  author={},
  booktitle={NeurIPS Datasets and Benchmarks Track},
  year={2025}
}
```

## License

MIT
