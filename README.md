# ToolBlind: Measuring AI Agent Reasoning Under Tool Absence with Trajectory Commitment

ToolBlind is a benchmark for evaluating whether AI agents can reason correctly when a required tool becomes unavailable mid-trajectory. Unlike existing tool-use benchmarks that test whether agents can *use* tools, ToolBlind tests whether agents can recognize tool *absence* and respond appropriately — by substituting, decomposing, or halting — rather than confabulating a path forward. The key independent variable is *trajectory commitment*: how many steps has the agent already completed before encountering the gap? Our hypothesis, supported by preliminary results, is that deeper prior commitment increases confabulation rates.

## Installation

```bash
git clone https://github.com/yourrepo/toolblind.git
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
