"""
Visualization functions for endorsement experiment analysis.

This module provides publication-ready plots for the endorsement framework.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def plot_bounds_comparison(ate, constant_term=None, figsize=(12, 8)):
    """
    Create comprehensive comparison plot of all estimation methods.
    
    Parameters:
    -----------
    ate : float
        Average treatment effect
    constant_term : float, optional
        Baseline support level
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Support Estimation Methods (ATE = {ate:.3f})', fontsize=16, fontweight='bold')
    
    # 1. Binary framework bounds
    p_max_binary = (ate + 1) / 2
    p_min_binary = max(0, p_max_binary)
    
    ax1 = axes[0, 0]
    ax1.barh(['Binary Framework'], [p_max_binary], color='skyblue', alpha=0.7)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel('Proportion Supporting Group')
    ax1.set_title('Binary Framework\n(No Baseline Info)')
    ax1.text(p_max_binary + 0.02, 0, f'{p_max_binary:.1%}', va='center', fontweight='bold')
    
    # 2. Baseline analysis (if available)
    ax2 = axes[0, 1]
    if constant_term is not None:
        p_baseline = constant_term + ate
        ax2.barh(['With Baseline'], [p_baseline], color='orange', alpha=0.7)
        ax2.text(p_baseline + 0.02, 0, f'{p_baseline:.1%}', va='center', fontweight='bold')
        ax2.set_title('Baseline Analysis\n(Exact Solution)')
    else:
        ax2.text(0.5, 0.5, 'Baseline info\nnot available', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12, style='italic')
        ax2.set_title('Baseline Analysis\n(Requires Constant Term)')
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Proportion Supporting Group')
    
    # 3. Continuous effects scenarios
    ax3 = axes[1, 0]
    
    # Different scenarios
    p_extreme = max(0, min(1, 1 + ate))  # Extreme opponent reaction
    p_symmetric = max(0, min(1, ate / (2 * 0.05) + 0.5))  # Small symmetric effects
    
    scenarios = ['Extreme\nOpponent', 'Symmetric\nEffects']
    values = [p_extreme, p_symmetric]
    colors = ['lightgreen', 'yellow']
    
    bars = ax3.bar(scenarios, values, color=colors, alpha=0.7)
    ax3.set_ylabel('Proportion Supporting Group')
    ax3.set_title('Continuous Effects\nScenarios')
    ax3.set_ylim(0, 1)
    
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Summary comparison
    ax4 = axes[1, 1]
    
    methods = ['Binary', 'Baseline', 'Extreme', 'Symmetric']
    estimates = [p_max_binary, 
                p_baseline if constant_term else np.nan,
                p_extreme, 
                p_symmetric]
    
    valid_methods = []
    valid_estimates = []
    colors_final = []
    
    color_map = {'Binary': 'skyblue', 'Baseline': 'orange', 'Extreme': 'lightgreen', 'Symmetric': 'yellow'}
    
    for method, est in zip(methods, estimates):
        if not np.isnan(est):
            valid_methods.append(method)
            valid_estimates.append(est)
            colors_final.append(color_map[method])
    
    bars = ax4.bar(valid_methods, valid_estimates, color=colors_final, alpha=0.7)
    ax4.set_ylabel('Proportion Supporting Group')
    ax4.set_title('Method Comparison')
    ax4.set_ylim(0, 1)
    
    for bar, val in zip(bars, valid_estimates):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_sensitivity_analysis(ate_range, constant_term, figsize=(10, 6)):
    """
    Plot sensitivity of support estimates to ATE values.
    
    Parameters:
    -----------
    ate_range : array-like
        Range of ATE values
    constant_term : float
        Baseline support level
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ate_values = np.array(ate_range)
    
    # Different methods
    binary_estimates = (ate_values + 1) / 2
    baseline_estimates = constant_term + ate_values
    extreme_estimates = np.clip(1 + ate_values, 0, 1)
    
    ax.plot(ate_values, binary_estimates, 'o-', label='Binary Framework', linewidth=2, markersize=6)
    ax.plot(ate_values, baseline_estimates, 's-', label='Baseline Analysis', linewidth=2, markersize=6)
    ax.plot(ate_values, extreme_estimates, '^-', label='Continuous (Extreme)', linewidth=2, markersize=6)
    
    # Add vertical line at actual ATE
    if len(ate_range) > 1:
        mid_ate = ate_range[len(ate_range)//2]
        ax.axvline(mid_ate, color='red', linestyle='--', alpha=0.7, label=f'Observed ATE = {mid_ate:.3f}')
    
    ax.set_xlabel('Average Treatment Effect')
    ax.set_ylabel('Estimated Proportion Supporting Group')
    ax.set_title('Sensitivity Analysis: Support Estimates vs. ATE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    return fig

def plot_effect_distribution(effects, ate, figsize=(10, 6)):
    """
    Plot distribution of individual treatment effects.
    
    Parameters:
    -----------
    effects : array-like
        Individual treatment effects
    ate : float
        Average treatment effect
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    ax1.hist(effects, bins=30, alpha=0.7, color='lightblue', edgecolor='black', density=True)
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='No Effect')
    ax1.axvline(np.mean(effects), color='orange', linestyle='-', linewidth=2, 
               label=f'Mean = {np.mean(effects):.4f}')
    ax1.axvline(ate, color='green', linestyle=':', linewidth=2, 
               label=f'Target ATE = {ate:.4f}')
    ax1.set_xlabel('Individual Treatment Effect')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Individual Effects')
    ax1.legend()
    
    # Cumulative distribution
    sorted_effects = np.sort(effects)
    cumulative_prop = np.arange(1, len(sorted_effects) + 1) / len(sorted_effects)
    
    ax2.plot(sorted_effects, cumulative_prop, linewidth=2, color='blue')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.fill_between(sorted_effects[sorted_effects > 0], 
                    cumulative_prop[sorted_effects > 0], 
                    alpha=0.3, color='green', label='Positive Effects')
    
    positive_prop = np.mean(effects > 0)
    ax2.text(0.02, 0.98, f'Positive Effects: {positive_prop:.1%}', 
            transform=ax2.transAxes, va='top', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Individual Treatment Effect')
    ax2.set_ylabel('Cumulative Proportion')
    ax2.set_title('Cumulative Distribution of Effects')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_policy_selection_guide(figsize=(12, 8)):
    """
    Create visualization for policy selection criteria.
    
    Parameters:
    -----------
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Policy Selection Guide for Endorsement Experiments', fontsize=16, fontweight='bold')
    
    # Example policies
    policies = {
        'Infrastructure': [0.85, 0.9, 0.8, 'Good'],
        'Vaccination': [0.88, 0.95, 0.7, 'Good'],
        'Military': [0.65, 0.2, 0.6, 'Poor'],
        'Healthcare': [0.70, 0.3, 0.5, 'Poor'],
        'Roads': [0.82, 0.85, 0.9, 'Good'],
        'Gun Control': [0.45, 0.1, 0.2, 'Poor']
    }
    
    # 1. Baseline support vs. ideology neutrality
    ax1 = axes[0, 0]
    for policy, (baseline, ideology, plausible, quality) in policies.items():
        color = 'green' if quality == 'Good' else 'red'
        ax1.scatter(baseline, ideology, s=100, c=color, alpha=0.7)
        ax1.annotate(policy, (baseline, ideology), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    
    # Add optimal region
    optimal_rect = Rectangle((0.8, 0.7), 0.1, 0.25, alpha=0.2, color='green')
    ax1.add_patch(optimal_rect)
    ax1.text(0.85, 0.82, 'Optimal\nRegion', ha='center', va='center', fontweight='bold')
    
    ax1.set_xlabel('Baseline Support')
    ax1.set_ylabel('Ideological Neutrality')
    ax1.set_title('Policy Selection: Baseline vs. Ideology')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # 2. Overall scoring
    ax2 = axes[0, 1]
    policy_names = list(policies.keys())
    overall_scores = []
    
    for policy, (baseline, ideology, plausible, quality) in policies.items():
        # Calculate overall score (weighted average)
        baseline_score = min(baseline/0.8, 1.0) if baseline < 0.8 else (1.0 if baseline <= 0.9 else max(0.1, 1-(baseline-0.9)/0.05))
        overall = 0.4 * baseline_score + 0.3 * ideology + 0.3 * plausible
        overall_scores.append(overall)
    
    colors = ['green' if score > 0.7 else 'orange' if score > 0.5 else 'red' for score in overall_scores]
    bars = ax2.bar(policy_names, overall_scores, color=colors, alpha=0.7)
    
    ax2.set_ylabel('Overall Score')
    ax2.set_title('Policy Quality Scores')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add score labels
    for bar, score in zip(bars, overall_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Criteria heatmap
    ax3 = axes[1, 0]
    criteria_data = np.array([[policies[p][i] for i in range(3)] for p in policy_names])
    
    im = ax3.imshow(criteria_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(['Baseline', 'Ideology', 'Plausible'])
    ax3.set_yticks(range(len(policy_names)))
    ax3.set_yticklabels(policy_names)
    ax3.set_title('Policy Criteria Heatmap')
    
    # Add text annotations
    for i in range(len(policy_names)):
        for j in range(3):
            ax3.text(j, i, f'{criteria_data[i, j]:.2f}', ha='center', va='center',
                    color='white' if criteria_data[i, j] < 0.5 else 'black', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('Score (0-1)')
    
    # 4. Guidelines text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    guidelines_text = """
    POLICY SELECTION GUIDELINES
    
    ✓ GOOD POLICIES:
    • 80-90% baseline support
    • High ideological neutrality
    • Plausible group endorsement
    • Concrete and specific
    
    ✗ AVOID:
    • Highly polarized issues
    • Near-universal support (>95%)
    • Implausible endorsements
    • Abstract principles
    
    EXAMPLES:
    Good: Infrastructure, health, education
    Poor: Military, taxes, gun control
    """
    
    ax4.text(0.05, 0.95, guidelines_text, transform=ax4.transAxes, fontsize=11,
            va='top', ha='left', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_blair_replication(figsize=(14, 10)):
    """
    Create comprehensive visualization for Blair et al. replication.
    
    Parameters:
    -----------
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    # Blair et al. parameters
    ate = -0.011
    constant = 0.8
    
    fig = plt.figure(figsize=figsize)
    
    # Create complex subplot layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main comparison plot (top row, spans 2 columns)
    ax_main = fig.add_subplot(gs[0, :2])
    
    methods = ['Binary\n(No Baseline)', 'Baseline\nAnalysis', 'Continuous\n(Extreme)', 'Continuous\n(Symmetric)']
    estimates = [
        (ate + 1) / 2,  # Binary
        constant + ate,  # Baseline
        max(0, min(1, 1 + ate)),  # Extreme
        max(0, min(1, ate / (2 * 0.05) + 0.5))  # Symmetric
    ]
    
    colors = ['skyblue', 'orange', 'lightgreen', 'yellow']
    bars = ax_main.bar(methods, estimates, color=colors, alpha=0.8, edgecolor='black')
    
    ax_main.set_ylabel('Proportion Supporting Militant Groups', fontsize=12)
    ax_main.set_title('Blair et al. (2013): Support Estimates by Method', fontsize=14, fontweight='bold')
    ax_main.set_ylim(0, 1)
    
    # Add percentage labels
    for bar, est in zip(bars, estimates):
        ax_main.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{est:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add horizontal line at 50%
    ax_main.axhline(0.5, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax_main.text(3.5, 0.52, 'Majority\nSupport', ha='center', va='bottom', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Treatment vs control comparison (top right)
    ax_comparison = fig.add_subplot(gs[0, 2])
    
    groups = ['Control', 'Treatment']
    values = [constant, constant + ate]
    colors_comp = ['lightblue', 'salmon']
    
    bars_comp = ax_comparison.bar(groups, values, color=colors_comp, alpha=0.8, edgecolor='black')
    ax_comparison.set_ylabel('Average Policy Support')
    ax_comparison.set_title('Group Averages')
    ax_comparison.set_ylim(0.75, 0.81)
    
    for bar, val in zip(bars_comp, values):
        ax_comparison.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                          f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add ATE annotation
    ax_comparison.annotate('', xy=(1, values[1]), xytext=(0, values[0]),
                          arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax_comparison.text(0.5, np.mean(values), f'ATE = {ate:.3f}', ha='center', va='center',
                      bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8), fontweight='bold')
    
    # Sensitivity analysis (middle row, spans all columns)
    ax_sensitivity = fig.add_subplot(gs[1, :])
    
    ate_range = np.linspace(-0.05, 0.05, 50)
    binary_sens = (ate_range + 1) / 2
    baseline_sens = constant + ate_range
    
    ax_sensitivity.plot(ate_range, binary_sens, 'o-', label='Binary Framework', linewidth=2, markersize=4)
    ax_sensitivity.plot(ate_range, baseline_sens, 's-', label='Baseline Analysis', linewidth=2, markersize=4)
    
    # Highlight actual ATE
    ax_sensitivity.axvline(ate, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'Observed ATE = {ate:.3f}')
    
    ax_sensitivity.set_xlabel('Average Treatment Effect')
    ax_sensitivity.set_ylabel('Estimated Support Proportion')
    ax_sensitivity.set_title('Sensitivity Analysis: How Support Estimates Change with ATE')
    ax_sensitivity.legend()
    ax_sensitivity.grid(True, alpha=0.3)
    ax_sensitivity.set_ylim(0, 1)
    
    # External validation (bottom left)
    ax_external = fig.add_subplot(gs[2, 0])
    ax_external.axis('off')
    
    validation_text = """
    EXTERNAL VALIDATION
    
    Pew Research (2015):
    • TTP explicit support: 9-14%
    • "Don't Know": 30-49%
    
    Our estimate: 78.9%
    
    Interpretation:
    High DK rates suggest many
    reluctant to express true
    preferences in direct polling.
    
    Endorsement experiments may
    capture hidden support.
    """
    
    ax_external.text(0.05, 0.95, validation_text, transform=ax_external.transAxes, fontsize=10,
                    va='top', ha='left', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8))
    
    # Policy context (bottom middle)
    ax_policy = fig.add_subplot(gs[2, 1])
    ax_policy.axis('off')
    
    policy_text = """
    BLAIR ET AL. POLICIES
    
    • Infrastructure development
    • Polio vaccination programs
    • Healthcare initiatives
    • Education funding
    
    CHARACTERISTICS:
    ✓ High baseline support (~80%)
    ✓ Ideologically neutral
    ✓ Plausible endorsements
    ✓ Public goods focus
    
    → Well-designed for detecting
       latent group attitudes
    """
    
    ax_policy.text(0.05, 0.95, policy_text, transform=ax_policy.transAxes, fontsize=10,
                  va='top', ha='left', fontfamily='monospace',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))
    
    # Key insights (bottom right)
    ax_insights = fig.add_subplot(gs[2, 2])
    ax_insights.axis('off')
    
    insights_text = """
    KEY INSIGHTS
    
    1. Baseline info crucial
       → Changes 49% to 79%
    
    2. Small negative ATE can
       mask majority support
    
    3. High baseline + small ATE
       = Many supporters
    
    4. Traditional interpretation
       ("negative sentiment")
       misleading without
       baseline context
    
    5. Policy design matters
       for interpretability
    """
    
    ax_insights.text(0.05, 0.95, insights_text, transform=ax_insights.transAxes, fontsize=10,
                    va='top', ha='left', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    return fig

# Example usage
if __name__ == "__main__":
    # Test visualizations
    print("Creating example visualizations...")
    
    # Blair et al. parameters
    ate = -0.011
    constant = 0.8
    
    # Create and show plots
    fig1 = plot_bounds_comparison(ate, constant)
    plt.show()
    
    fig2 = plot_blair_replication()
    plt.show()
    
    print("Visualization module loaded successfully!")