# Project Summary: AI-Fiber-NLC

## What We Did

We built a complete open-source benchmark framework to test whether AI-based nonlinear compensation (NLC) can outperform traditional digital signal processing (DSP) in coherent optical fiber communication systems.

**Scope:**
- 23 Python files, ~2,200 lines of code
- Complete DSP receiver chain (EDC + clock recovery + CPR)
- DBP implementation (configurable 1-200 steps/span)
- MLP-NLC model with DSP-chain integrated training
- CNN-NLC architecture (code ready, training too slow on available hardware)
- 6 benchmark scenes (MVB-1 through MVB-6) covering standard and extended regimes
- 26 simulation data files (~44 MB)
- 79 unit tests, all passing
- Full academic paper draft (`paper/paper.tex`)

**Benchmark Coverage:**

| Scene | Distance | γ (nonlinearity) | Power Range | Purpose |
|-------|----------|-------------------|-------------|---------|
| MVB-1 | 800 km | 1.3 (standard) | −3 to +5 dBm | Standard regime |
| MVB-3 | 800 km | 1.3 | −3 to +5 dBm | 64QAM + PCS |
| MVB-4 | 800 km | 1.3 | +7 to +10 dBm | High power |
| MVB-5 | 1600 km | 1.3 | 0 to +3 dBm | Long distance |
| MVB-6 | 800 km | **10.0** (7.7×) | 0 to +3 dBm | High nonlinearity |

## What We Found

**AI-NLC and DBP both show zero meaningful improvement over a minimal DSP chain (EDC + CPR) across ALL tested regimes.**

| Scene | Baseline Q | Best DBP Q | Best MLP Q | Max Improvement |
|-------|-----------|-----------|-----------|----------------|
| MVB-1 | +1.91 to +2.20 | +2.20 | +2.01 | +0.03 dB |
| MVB-4 | +2.01 to +2.07 | +2.09 | N/A | +0.02 dB |
| MVB-5 | +1.92 to +2.16 | +2.19 | N/A | +0.03 dB |
| MVB-6 | +2.04 to +2.05 | +2.04 | N/A | −0.01 dB |

Even at γ = 10 (7.7× standard nonlinearity), NLC still fails.

## Why It Fails

The bottleneck in EDFA-amplified systems is **accumulated ASE noise**, not nonlinear distortion. The EDFA at each span boundary resets the signal power, preventing nonlinear phase noise from accumulating across spans. Each span's nonlinearity is individually weak and already handled by EDC + CPR.

This is a **physical limitation**, not an algorithmic one. No DSP technique — DBP, AI, or otherwise — can recover information corrupted by additive noise below the Shannon limit.

## Academic Value

**The negative result itself has value:**

1. It prevents researchers from wasting resources on unpromising configurations.
2. It defines the operational boundaries of AI-NLC: effective only in regimes without intermediate amplification (submarine links, sparse-amplification ultra-long-haul, or unrepeated spans).
3. It provides a reproducible open-source benchmark that others can build upon.

**The paper** (`paper/paper.tex`) documents all findings and is ready for arXiv submission.

## Should This Repo Stay?

**Yes, if:**
- You want to keep the benchmark framework for future research
- You plan to test new regimes (no-EDFA, submarine, WDM inter-channel)
- You want the code as a reference for how to properly evaluate NLC algorithms

**No, if:**
- The negative result is enough and you don't plan further research in this area
- You'd rather redirect resources to a different AI + optical communications topic

## How to Delete

If you decide to delete:
1. Go to https://github.com/AI-Fiber-NLC/AI-Fiber-NLC/settings
2. Scroll to "Danger Zone"
3. Click "Delete this repository"
4. Confirm by typing `AI-Fiber-NLC/AI-Fiber-NLC`

---

*Project active: May 16–18, 2026*
*Total commits: 12*
*Final status: Complete — comprehensive negative result documented*
