## Inferring Support From Endorsement Experiments

## Manuscript

* [Manuscript (pdf and .tex)](ms/)

## The Problem: Endorsement Experiments Are Underidentified

Blair et al. (2013) conducted endorsement experiments in Pakistan and concluded:

> "We find that Pakistanis in general are **weakly negative** toward Islamist militant organizations."

> "The coefficients are negative and statistically significant...suggesting that **Pakistanis hold militant groups in low regard**."

**This interpretation is incomplete.**

The sign of the average treatment effect (ATE) tells us the balance of switchers, not the level of support. With 80% baseline policy support and a -0.011 treatment effect, implied support for militant groups ranges from **39% to 99%** depending on assumptions about individual switching behavior:

| Model | Assumption | Implied Support |
|-------|------------|-----------------|
| Symmetric effects | Both groups shift ±0.05 | 39% |
| Binary switching | All supporters → 1, opponents → 0 | 79% |
| Extreme reaction | Opponents react maximally | 99% |

No model supports Blair et al.'s characterization of "low regard." But the precise level of support is sensitive to assumptions the data cannot adjudicate.

See the [manuscript](ms/endorsement.pdf) for the full analysis.

## Overview

Endorsement experiments identify the ATE under standard exclusion restrictions, but the ATE alone does not identify the proportion of supporters. This toolkit provides:

- **Bounds under different behavioral models** (binary switching, symmetric effects, etc.)
- **Baseline support analysis** using constant terms from regressions
- **Variance tests** for the binary switching assumption
- **Sensitivity analysis** across the range of plausible models
- **Visualization tools** for publication-ready figures

## Quick Start

### Option 1: Quick Analysis Script
```python
# Modify these parameters in endorsement_quick_analysis.py
YOUR_ATE = -0.011              # Your treatment effect
YOUR_CONSTANT = 0.8            # Your baseline support
YOUR_SAMPLE_SIZE = 1500        # Your sample size

# Run analysis
python3 endorsement_quick_analysis.py
```

### Option 2: Interactive Analysis
```python
from endorsement_analysis_notebook import EndorsementAnalyzer

# Initialize with your experimental results
analyzer = EndorsementAnalyzer(ate=-0.011, constant_term=0.8)

# Get estimates under different models
baseline = analyzer.baseline_support_analysis()
print(f"Binary switching estimate: {baseline['proportion_supporters']:.1%}")

binary = analyzer.binary_framework_bounds()
print(f"Binary (no baseline) max: {binary['p_max']:.1%}")
```

## File Structure

```
├── ms/endorsement.tex                          # Manuscript with full theoretical framework
├── scripts/blair_reinterpretation.ipynb        # Blair et al. critique notebook
├── scripts/endorsement_analysis_notebook.py    # Main analysis classes
├── scripts/endorsement_utils.py                # Utility functions
├── scripts/endorsement_viz.py                  # Visualization tools
├── scripts/endorsement_quick_analysis.py       # Quick analysis script
├── scripts/generate_tables_and_figures.py      # Generates tabs/ and figs/
├── tabs/                                       # LaTeX tables for manuscript
├── figs/                                       # Figures for manuscript
└── README.md
```

## Key Methods

### 1. Binary Switching
**Assumption:** All supporters end up at 1, all opponents at 0 in treatment.
```python
proportion_supporters = constant_term + ate  # = 0.80 - 0.011 = 78.9%
```
**Testable implication:** Treatment group variance should equal p(1-p) ≈ 0.166. If observed variance is much lower, binary switching is rejected.

### 2. Symmetric Effects
**Assumption:** Supporters shift by +d, opponents by -d.
```python
p = ate / (2 * d) + 0.5  # With d=0.05: -0.011/0.1 + 0.5 = 39%
```

### 3. Extreme Opponent Reaction
**Assumption:** Opponents react maximally (δ = -1), supporters unaffected (δ = 0).
```python
p = 1 + ate  # = 1 - 0.011 = 98.9%
```

### 4. Without Baseline Information
If only the ATE is available (no constant term):
```python
p_max = (ate + 1) / 2  # = 49.5%
```
This is far less informative than using baseline support.

## The Identification Problem

The ATE decomposes as:
```
β = p × δ⁺ + (1-p) × δ⁻
```

Three unknowns (p, δ⁺, δ⁻), one equation. The system is underidentified without restrictions on switching behavior. Different assumptions yield very different estimates:

| Without Baseline | With Baseline (α=0.8) | Model |
|------------------|----------------------|-------|
| Max 49.5% | 39% | Symmetric (d=0.05) |
| Max 49.5% | 79% | Binary switching |
| Max 49.5% | 99% | Extreme reaction |

**Key insight:** Baseline information is crucial and routinely ignored.

## Guidance for Practitioners

1. **Report treatment group averages, not just ATEs.** Under binary switching, the treatment average directly estimates the supporter proportion.

2. **Do not equate the ATE sign with "sentiment."** The sign tells you which switching force dominates, not whether the group is popular.

3. **Discuss identifying assumptions explicitly.** Present sensitivity analyses across plausible behavioral models.

4. **Test the binary switching assumption.** Compare treatment variance to p(1-p). Large discrepancies indicate binary switching is inappropriate.

## Policy Selection Guidelines

Optimal policies for endorsement experiments should have:
- **Baseline support: 80-90%** (room for opponents to move)
- **Ideological neutrality** (public goods, not partisan issues)
- **Plausible endorsement** (group could realistically support it)
- **Avoid ceiling effects** (not >95% support)

### Good Policies
- Infrastructure development
- Public health programs (polio vaccination)
- Education funding

### Poor Policies
- Military spending (ideologically polarized)
- Taxation policy (partisan)
- Gun control (low baseline, polarized)

## Visualization

```python
from endorsement_viz import *

# Compare estimates across methods
fig1 = plot_bounds_comparison(ate=-0.011, constant_term=0.8)

# Sensitivity to ATE
fig2 = plot_sensitivity_analysis(np.linspace(-0.05, 0.05, 50), constant_term=0.8)
```

## Dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy
```

## Citation

```
Sood, Gaurav. (2025). Inferring Support from Endorsement Experiments.
```

## Key Takeaways

1. **Endorsement experiments are underidentified** for the proportion of supporters without behavioral assumptions
2. **Baseline support information is crucial** and changes estimates dramatically
3. **The ATE sign does not measure "sentiment"** - it measures the balance of switchers
4. **Report ranges, not point estimates** unless you can defend a specific behavioral model
5. **Binary switching is testable** via the variance implication - and often rejected
