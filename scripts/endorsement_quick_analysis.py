#!/usr/bin/env python3
"""
Quick Endorsement Experiment Analysis

A simple script for researchers to quickly analyze their endorsement experiments.
Just modify the parameters below and run to get comprehensive analysis.

Usage:
    python endorsement_quick_analysis.py

Author: Gaurav Sood
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def quick_endorsement_analysis(ate, constant_term=None, sample_size=None, 
                             group_name="endorsing group", policy_name="the policy"):
    """
    Perform quick analysis of endorsement experiment results.
    
    Parameters:
    -----------
    ate : float
        Average treatment effect (beta coefficient from regression)
    constant_term : float, optional
        Baseline support level (alpha coefficient from regression)
    sample_size : int, optional
        Sample size (for power calculations)
    group_name : str
        Name of the endorsing group (for output)
    policy_name : str
        Name/description of the policy (for output)
    
    Returns:
    --------
    dict: Comprehensive analysis results
    """
    
    print("=" * 70)
    print(f"ENDORSEMENT EXPERIMENT ANALYSIS")
    print("=" * 70)
    print(f"Group: {group_name}")
    print(f"Policy: {policy_name}")
    print(f"Average Treatment Effect: {ate:.4f}")
    if constant_term is not None:
        print(f"Baseline Support: {constant_term:.3f}")
    if sample_size is not None:
        print(f"Sample Size: {sample_size}")
    print()
    
    results = {}
    
    # 1. Binary Framework Analysis (without baseline)
    print("1. BINARY FRAMEWORK BOUNDS (No Baseline Information)")
    print("-" * 50)
    
    p_max_binary = (ate + 1) / 2
    p_min_binary = max(0, p_max_binary)
    
    print(f"   Maximum supporters: {p_max_binary:.1%}")
    print(f"   Minimum supporters: {p_min_binary:.1%}")
    print(f"   Interpretation: Between {p_min_binary:.1%} and {p_max_binary:.1%} could support {group_name}")
    print()
    
    results['binary'] = {
        'p_max': p_max_binary,
        'p_min': p_min_binary
    }
    
    # 2. Baseline Analysis (if available)
    if constant_term is not None:
        print("2. BASELINE SUPPORT ANALYSIS (With Constant Term)")
        print("-" * 50)
        
        treatment_avg = constant_term + ate
        control_avg = constant_term
        proportion_supporters = treatment_avg
        
        print(f"   Control group average: {control_avg:.1%}")
        print(f"   Treatment group average: {treatment_avg:.1%}")
        print(f"   Exact proportion of supporters: {proportion_supporters:.1%}")
        
        if proportion_supporters > 0.5:
            print(f"   → MAJORITY ({proportion_supporters:.1%}) supports {group_name}")
        else:
            print(f"   → MINORITY ({proportion_supporters:.1%}) supports {group_name}")
        
        print(f"   → Despite negative ATE, {proportion_supporters:.1%} are actually supporters!")
        print()
        
        results['baseline'] = {
            'proportion_supporters': proportion_supporters,
            'treatment_avg': treatment_avg,
            'control_avg': control_avg
        }
        
        # Dramatic difference highlight
        difference = proportion_supporters - p_max_binary
        print(f"   🔍 KEY INSIGHT: Baseline info changes estimate by {difference:+.1%}!")
        print(f"      Binary bound: {p_max_binary:.1%} vs Baseline estimate: {proportion_supporters:.1%}")
        print()
    
    # 3. Continuous Effects Scenarios
    print("3. CONTINUOUS EFFECTS SCENARIOS")
    print("-" * 50)
    
    # Extreme opponent reaction
    p_extreme = max(0, min(1, 1 + ate))
    print(f"   Extreme scenario (opponents 1→0, supporters unchanged): {p_extreme:.1%}")
    
    # Symmetric small effects
    small_effect = 0.05
    p_symmetric = max(0, min(1, ate / (2 * small_effect) + 0.5))
    print(f"   Symmetric small effects (±{small_effect}): {p_symmetric:.1%}")
    
    # High baseline scenario (if baseline available)
    if constant_term is not None and constant_term > 0.7:
        # Assume opponents drop modestly
        modest_drop = 0.1
        implied_opponents = (control_avg - treatment_avg) / modest_drop
        p_realistic = 1 - implied_opponents
        print(f"   Realistic scenario (opponents drop {modest_drop:.1f}): {p_realistic:.1%}")
        
        results['continuous'] = {
            'p_extreme': p_extreme,
            'p_symmetric': p_symmetric,
            'p_realistic': p_realistic
        }
    else:
        results['continuous'] = {
            'p_extreme': p_extreme,
            'p_symmetric': p_symmetric
        }
    
    print()
    
    # 4. Interpretation and Recommendations
    print("4. INTERPRETATION & RECOMMENDATIONS")
    print("-" * 50)
    
    if constant_term is not None:
        main_estimate = proportion_supporters
        method = "baseline analysis"
    else:
        main_estimate = p_max_binary
        method = "binary bounds"
    
    if main_estimate > 0.7:
        interpretation = f"STRONG MAJORITY support for {group_name}"
        recommendation = "Consider that group endorsements might actually help policy adoption"
    elif main_estimate > 0.5:
        interpretation = f"MAJORITY support for {group_name}"
        recommendation = "Group has more support than negative ATE suggests"
    elif main_estimate > 0.3:
        interpretation = f"SUBSTANTIAL MINORITY support for {group_name}"
        recommendation = "Significant latent support exists despite negative sentiment"
    else:
        interpretation = f"LIMITED support for {group_name}"
        recommendation = "Group endorsement clearly hurts policy support"
    
    print(f"   Best estimate ({method}): {main_estimate:.1%}")
    print(f"   Interpretation: {interpretation}")
    print(f"   Recommendation: {recommendation}")
    print()
    
    # 5. Policy Assessment
    if constant_term is not None:
        print("5. POLICY DESIGN ASSESSMENT")
        print("-" * 50)
        
        if constant_term >= 0.8 and constant_term <= 0.9:
            policy_quality = "EXCELLENT"
        elif constant_term >= 0.7 and constant_term <= 0.95:
            policy_quality = "GOOD"
        elif constant_term >= 0.95:
            policy_quality = "CEILING RISK"
        else:
            policy_quality = "SUBOPTIMAL"
        
        print(f"   Baseline support level: {constant_term:.1%} ({policy_quality})")
        
        if policy_quality == "EXCELLENT":
            print("   → Optimal for endorsement experiments (80-90% baseline)")
        elif policy_quality == "GOOD":
            print("   → Adequate for endorsement experiments")
        elif policy_quality == "CEILING RISK":
            print("   → May have ceiling effects (too popular)")
        else:
            print("   → Consider policies with higher baseline support")
        
        print()
    
    # 6. Statistical Power (if sample size available)
    if sample_size is not None:
        print("6. STATISTICAL POWER ASSESSMENT")
        print("-" * 50)
        
        # Rough power calculation
        effect_size = abs(ate)
        std_estimate = 0.3  # Rough estimate for binary outcomes
        cohens_d = effect_size / std_estimate
        
        if cohens_d > 0.8:
            power_assessment = "LARGE effect - well powered"
        elif cohens_d > 0.5:
            power_assessment = "MEDIUM effect - adequately powered"
        elif cohens_d > 0.2:
            power_assessment = "SMALL effect - may be underpowered"
        else:
            power_assessment = "VERY SMALL effect - likely underpowered"
        
        print(f"   Sample size: {sample_size}")
        print(f"   Effect size: {effect_size:.4f}")
        print(f"   Assessment: {power_assessment}")
        print()
    
    results['interpretation'] = {
        'main_estimate': main_estimate,
        'method': method,
        'interpretation': interpretation,
        'recommendation': recommendation
    }
    
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    return results

def create_quick_visualization(results, ate, constant_term=None, save_file=None):
    """
    Create a quick summary visualization.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left plot: Method comparison
    methods = ['Binary\nBounds']
    estimates = [results['binary']['p_max']]
    colors = ['skyblue']
    
    if 'baseline' in results:
        methods.append('Baseline\nAnalysis')
        estimates.append(results['baseline']['proportion_supporters'])
        colors.append('orange')
    
    if 'continuous' in results and 'p_extreme' in results['continuous']:
        methods.append('Continuous\nExtreme')
        estimates.append(results['continuous']['p_extreme'])
        colors.append('lightgreen')
    
    bars = axes[0].bar(methods, estimates, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('Proportion Supporting Group')
    axes[0].set_title(f'Support Estimates (ATE = {ate:.3f})')
    axes[0].set_ylim(0, 1)
    
    # Add percentage labels
    for bar, est in zip(bars, estimates):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{est:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Add 50% line
    axes[0].axhline(0.5, color='red', linestyle='--', alpha=0.7)
    axes[0].text(len(methods)-0.5, 0.52, 'Majority', ha='center', va='bottom')
    
    # Right plot: Treatment vs Control (if baseline available)
    if constant_term is not None:
        groups = ['Control', 'Treatment']
        values = [constant_term, constant_term + ate]
        
        bars = axes[1].bar(groups, values, color=['lightblue', 'salmon'], 
                          alpha=0.8, edgecolor='black')
        axes[1].set_ylabel('Average Policy Support')
        axes[1].set_title('Control vs Treatment Groups')
        
        for bar, val in zip(bars, values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add ATE arrow
        axes[1].annotate('', xy=(1, values[1]), xytext=(0, values[0]),
                        arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        axes[1].text(0.5, np.mean(values), f'ATE = {ate:.3f}', ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='yellow'), fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'Baseline information\nnot available', 
                    ha='center', va='center', transform=axes[1].transAxes, 
                    fontsize=12, style='italic')
        axes[1].set_title('Control vs Treatment')
    
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_file}")
    
    return fig

# =============================================================================
# MAIN ANALYSIS - MODIFY THESE PARAMETERS FOR YOUR EXPERIMENT
# =============================================================================

if __name__ == "__main__":
    
    # YOUR EXPERIMENT PARAMETERS - MODIFY THESE
    YOUR_ATE = -0.011              # Your average treatment effect (beta coefficient)
    YOUR_CONSTANT = 0.8            # Your baseline support (alpha coefficient) - set to None if unknown
    YOUR_SAMPLE_SIZE = 1500        # Your total sample size - set to None if unknown
    YOUR_GROUP_NAME = "militant groups"      # Name of endorsing group
    YOUR_POLICY_NAME = "infrastructure and health policies"  # Policy description
    
    # RUN ANALYSIS
    print("Starting endorsement experiment analysis...")
    print()
    
    # Perform analysis
    results = quick_endorsement_analysis(
        ate=YOUR_ATE,
        constant_term=YOUR_CONSTANT,
        sample_size=YOUR_SAMPLE_SIZE,
        group_name=YOUR_GROUP_NAME,
        policy_name=YOUR_POLICY_NAME
    )
    
    # Create visualization
    print("Creating visualization...")
    fig = create_quick_visualization(results, YOUR_ATE, YOUR_CONSTANT, 
                                   save_file="endorsement_analysis.png")
    plt.show()
    
    # Save detailed results to CSV
    if results.get('baseline'):
        summary_data = {
            'Method': ['Binary Framework', 'Baseline Analysis'],
            'Estimate': [results['binary']['p_max'], results['baseline']['proportion_supporters']],
            'Interpretation': [f"Up to {results['binary']['p_max']:.1%} support", 
                             f"Exactly {results['baseline']['proportion_supporters']:.1%} support"]
        }
    else:
        summary_data = {
            'Method': ['Binary Framework'],
            'Estimate': [results['binary']['p_max']],
            'Interpretation': [f"Up to {results['binary']['p_max']:.1%} support"]
        }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('endorsement_results.csv', index=False)
    print("Results saved to: endorsement_results.csv")
    
    # Final recommendation
    print("\n" + "🎯 FINAL RECOMMENDATION:")
    print("-" * 30)
    if results.get('baseline') and results['baseline']['proportion_supporters'] > 0.5:
        print("Despite the negative ATE, MAJORITY support exists for the endorsing group.")
        print("The baseline support level reveals that small negative effects can mask")
        print("substantial underlying support. Consider this in policy communications.")
    elif results['binary']['p_max'] > 0.4:
        print("Substantial support may exist despite negative average effects.")
        print("Consider collecting baseline support data for more precise estimates.")
    else:
        print("Limited support for the endorsing group is indicated.")
        print("The negative endorsement effect appears to reflect genuine opposition.")
    
    print("\nAnalysis complete! Check the generated files for detailed results.")