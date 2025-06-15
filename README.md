## Inferring Support From Endorsement Experiments

## Manuscript

* [Manuscript (pdf and .tex)](ms/)

## Overview

Endorsement experiments measure how group endorsements affect policy support, but interpreting the results to infer actual support for the endorsing group is complex. This toolkit provides:

- **Theoretical bounds** on supporter proportions under different assumptions
- **Baseline support analysis** using constant terms from regressions
- **Continuous effects modeling** for non-binary outcomes
- **Policy selection guidance** for optimal experimental design
- **Visualization tools** for publication-ready figures

## Quick Start

### Option 1: Quick Analysis Script
For immediate analysis of your results:

```python
# Modify these parameters in endorsement_quick_analysis.py
YOUR_ATE = -0.011              # Your treatment effect
YOUR_CONSTANT = 0.8            # Your baseline support (if available)
YOUR_SAMPLE_SIZE = 1500        # Your sample size
YOUR_GROUP_NAME = "militant groups"
YOUR_POLICY_NAME = "infrastructure policies"

# Run analysis
python endorsement_quick_analysis.py
```

### Option 2: Interactive Analysis
Use the main notebook for detailed exploration:

```python
from endorsement_analysis_notebook import EndorsementAnalyzer

# Initialize with your experimental results
analyzer = EndorsementAnalyzer(ate=-0.011, constant_term=0.8)

# Get baseline analysis (most important!)
results = analyzer.baseline_support_analysis()
print(f"Proportion supporting group: {results['proportion_supporters']:.1%}")

# Compare with binary framework
binary_results = analyzer.binary_framework_bounds()
print(f"Binary framework max: {binary_results['p_max']:.1%}")
```

## File Structure

```
├── scripts/endorsement_analysis_notebook.py    # Main analysis classes and functions
├── scripts/endorsement_utils.py                # Utility functions and calculations  
├── scripts/endorsement_viz.py                  # Visualization tools
├── scripts/endorsement_quick_analysis.py       # Quick analysis script
└── README.md                          # This file
```

## Key Methods

### 1. Binary Framework
**When to use:** Basic bounds without baseline information
```python
p_max = (ate + 1) / 2
```

### 2. Baseline Support Analysis ⭐ **MOST IMPORTANT**
**When to use:** When you have the constant term from your regression
```python
proportion_supporters = constant_term + ate
```

**Why it matters:** This can dramatically change interpretation. In Blair et al. (2013):
- Binary framework: Max 49.5% supporters
- Baseline analysis: **78.9% supporters** 

### 3. Continuous Effects
**When to use:** Non-binary outcomes (e.g., 5-point scales)
- Extreme scenario: `p_max = 1 + ate` (if opponents react maximally)
- Symmetric effects: `p = ate/(2×effect_size) + 0.5`

### 4. Policy Selection Scoring
**When to use:** Designing new endorsement experiments
```python
score = analyzer.policy_selection_score(
    baseline_support=0.85,     # 80-90% optimal
    ideological_score=0.9,     # Higher = more neutral
    plausibility_score=0.8     # Higher = more believable
)
```

## Real-World Example: Blair et al. (2013)

**Study:** Pakistani attitudes toward militant groups via policy endorsements
**Results:** ATE = -0.011, Baseline = 0.8

**Traditional interpretation:** "Negative sentiment toward militant groups"
**Our analysis:** "78.9% actually support militant groups despite negative ATE"

```python
# Replicate Blair et al. analysis
results = blair_replication()

# Key insight: Small negative ATE + high baseline = majority support
# This happens when popular policies get endorsed by controversial groups
```

## Understanding the Results

### When Baseline Info Changes Everything

| Scenario | Without Baseline | With Baseline (α=0.8) | Interpretation |
|----------|------------------|----------------------|----------------|
| ATE = -0.01 | Max 49.5% support | **78.9% support** | Majority support despite negative ATE |
| ATE = -0.05 | Max 47.5% support | **75% support** | Still majority support |
| ATE = +0.01 | Max 50.5% support | **80.1% support** | Strong majority support |

### Why This Happens
- **High baseline support** (α = 0.8) means 80% support policy in control
- **Small negative ATE** (β = -0.011) means treatment drops to 78.9%  
- Under **binary switching**: treatment group average = proportion of supporters
- **Result:** 78.9% are supporters who always support when their group endorses

## Policy Selection Guidelines

### ✅ Good Policies for Endorsement Experiments
- **Infrastructure development** (baseline ~85%, neutral, plausible)
- **Public health programs** (baseline ~88%, neutral, plausible)
- **Education funding** (baseline ~82%, neutral, plausible)

### ❌ Poor Policies
- **Military spending** (ideologically polarized)
- **Universal healthcare** (ideologically polarized)  
- **Gun control** (low baseline, polarized)

### Optimal Characteristics
1. **Baseline support: 80-90%** (room for opponents to move)
2. **Ideologically neutral** (public goods, not partisan issues)
3. **Plausible endorsement** (group could realistically support it)
4. **Avoid ceiling effects** (not >95% support)

## Advanced Usage

### Simulation for Heterogeneous Effects
```python
# Simulate individual-level heterogeneity
sim_results = analyzer.simulate_heterogeneous_effects(
    effect_distribution='normal',
    std=0.05
)
print(f"Positive shifters: {sim_results['proportion_positive']:.1%}")
```

### Sensitivity Analysis
```python
# Test robustness across ATE range
ate_range = np.linspace(-0.05, 0.05, 50)
sensitivity_df = sensitivity_analysis(ate_range, constant_term=0.8)
```

### Optimization (for continuous outcomes)
```python
# Find maximum positive shifters using Hungarian algorithm
control_responses = generate_realistic_responses(100)['control']
treatment_responses = generate_realistic_responses(100)['treatment'] 
optimal_result = hungarian_optimization(control_responses, treatment_responses)
```

## Visualization Examples

```python
from endorsement_viz import *

# Comprehensive bounds comparison
fig1 = plot_bounds_comparison(ate=-0.011, constant_term=0.8)

# Blair et al. replication with all details
fig2 = plot_blair_replication()

# Policy selection guidance
fig3 = plot_policy_selection_guide()

# Sensitivity analysis
fig4 = plot_sensitivity_analysis(np.linspace(-0.05, 0.05, 50), constant_term=0.8)
```

## Dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy
```

## Citation

If you use this toolkit, please cite:

```
Sood, Gaurav. (2025). Inferring Support from Endorsement Experiments.
```

## Key Takeaways

1. **Baseline support information is crucial** - it can completely change your interpretation
2. **Small negative ATEs don't necessarily mean opposition** when baseline support is high
3. **Policy selection matters** - choose policies with 80-90% baseline support and ideological neutrality
4. **Binary switching models provide clean theoretical foundations** before extending to continuous outcomes
5. **External validation is important** - compare your estimates with other measures of support

## Common Mistakes to Avoid

❌ **Interpreting ATE alone** without considering baseline support
❌ **Using polarized policies** that confound group and ideological effects  
❌ **Assuming symmetric effects** without theoretical justification
❌ **Ignoring sample size** in power calculations
❌ **Focusing only on statistical significance** rather than substantive interpretation
