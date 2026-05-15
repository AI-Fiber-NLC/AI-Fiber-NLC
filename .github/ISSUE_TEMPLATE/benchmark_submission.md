---
name: Benchmark Submission
about: Submit a Nonlinear Compensation model benchmark result
title: '[Benchmark] Model Name - MVB-X'
labels: ['benchmark', 'submission']
---

## Model Information

- **Model Name:** 
- **Model Type:** (e.g., MLP, CNN, Transformer, KAN, DBP)
- **Author:** 
- **GitHub Handle:** 
- **Date:** YYYY-MM-DD

## Benchmark Scenario

- [ ] MVB-1 (Single-Pol 16QAM, 800km)
- [ ] MVB-2 (DP-16QAM, 800km)
- [ ] MVB-3 (64QAM + PCS, 800km)

## Training Configuration

```yaml
# Paste your training config (learning rate, epochs, batch size, etc.)
```

## Results

| Metric | Value |
|--------|-------|
| Q-factor (dB) | |
| Baseline Q-factor (dB) | |
| Q-factor Improvement (dB) | |
| DBP Q-factor (dB) | |
| Inference FLOPs per Symbol | |
| DBP FLOPs per Symbol | |
| FLOPs Ratio | |
| Training Time | |
| Hardware Used | |

## Contribution Signature

- [ ] I have generated a contribution signature using `tools/sign.py`
- [ ] The signature file (`contribution.sig`) is attached or included in this PR

## Additional Notes

<!-- Any observations, failure cases, or insights about your model -->

## Checklist

- [ ] Results obtained using the official benchmark script (`benchmark/run.py`)
- [ ] Training seed matches the scenario configuration
- [ ] Tested at minimum 3 power points within the specified range
- [ ] Code follows the project coding style
- [ ] Model weights are available (via Hugging Face or included in PR)
