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

## Reproducing the Paper

```bash
python3 scripts/generate_tables_and_figures.py
cd ms && pdflatex endorsement.tex
```

## File Structure

```
├── ms/endorsement.tex                      # Manuscript
├── scripts/generate_tables_and_figures.py  # Generates tabs/ and figs/
├── tabs/                                   # LaTeX tables for manuscript
├── figs/                                   # Figures for manuscript
└── README.md
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
