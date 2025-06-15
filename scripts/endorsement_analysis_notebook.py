"""
Inferring Support from Endorsement Experiments: Analysis Notebook

This notebook implements the methods from "Inferring Support from Endorsement Experiments: 
A Computational Perspective" by Gaurav Sood.

Author: Gaurav Sood
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EndorsementAnalyzer:
    """
    Main class for analyzing endorsement experiments and inferring support levels.
    """
    
    def __init__(self, ate, constant_term=None, sample_size=None):
        """
        Initialize analyzer with experimental results.
        
        Parameters:
        -----------
        ate : float
            Average treatment effect (beta coefficient)
        constant_term : float, optional
            Baseline support level (alpha coefficient)
        sample_size : int, optional
            Sample size for simulations
        """
        self.ate = ate
        self.constant_term = constant_term
        self.sample_size = sample_size or 1000
        
    def binary_framework_bounds(self):
        """
        Calculate bounds under binary switching framework (without baseline info).
        
        Returns:
        --------
        dict: Contains max and min proportions of supporters
        """
        p_max = (self.ate + 1) / 2
        p_min = max(0, (self.ate + 1) / 2)
        
        return {
            'p_max': p_max,
            'p_min': p_min,
            'framework': 'binary_no_baseline'
        }
    
    def baseline_support_analysis(self):
        """
        Calculate exact support proportion using baseline support information.
        
        Returns:
        --------
        dict: Contains exact proportion and interpretation
        """
        if self.constant_term is None:
            raise ValueError("Constant term required for baseline analysis")
            
        # Under complete binary switching: treatment average = proportion of supporters
        treatment_avg = self.constant_term + self.ate
        p_supporters = treatment_avg
        
        return {
            'proportion_supporters': p_supporters,
            'treatment_average': treatment_avg,
            'control_average': self.constant_term,
            'interpretation': f"{p_supporters:.1%} of respondents support the endorsing group"
        }
    
    def continuous_bounds(self, max_effect=1.0):
        """
        Calculate bounds under continuous effects framework.
        
        Parameters:
        -----------
        max_effect : float
            Maximum individual effect size
            
        Returns:
        --------
        dict: Contains various bound scenarios
        """
        # Extreme opponent reaction (supporters unaffected)
        p_max_extreme = 1 + self.ate  # Assumes delta_opp = -1, delta_supp = 0
        
        # Symmetric small effects
        effect_size = max_effect / 4  # Conservative effect size
        p_symmetric = self.ate / (2 * effect_size) + 0.5
        
        # High baseline scenario (if baseline available)
        if self.constant_term is not None:
            # Opponents drop modestly, supporters unchanged
            modest_drop = 0.1
            p_realistic = (self.constant_term - (self.constant_term + self.ate)) / modest_drop
        else:
            p_realistic = None
            
        return {
            'p_max_extreme': max(0, min(1, p_max_extreme)),
            'p_symmetric': max(0, min(1, p_symmetric)),
            'p_realistic': p_realistic,
            'effect_size_assumed': effect_size
        }
    
    def simulate_heterogeneous_effects(self, effect_distribution='normal', **kwargs):
        """
        Simulate heterogeneous individual effects and count positive shifters.
        
        Parameters:
        -----------
        effect_distribution : str
            'normal', 'beta', or 'uniform'
        **kwargs : dict
            Distribution parameters
            
        Returns:
        --------
        dict: Simulation results
        """
        n = self.sample_size
        
        # Generate individual effects with constraint that mean = ATE
        if effect_distribution == 'normal':
            std = kwargs.get('std', 0.05)
            effects = np.random.normal(self.ate, std, n)
            # Truncate to [-1, 1] and adjust to maintain ATE
            effects = np.clip(effects, -1, 1)
            effects = effects - np.mean(effects) + self.ate
            
        elif effect_distribution == 'beta':
            # Beta distribution rescaled to [-1, 1]
            alpha = kwargs.get('alpha', 2)
            beta_param = kwargs.get('beta', 2)
            raw_effects = 2 * np.random.beta(alpha, beta_param, n) - 1
            effects = raw_effects - np.mean(raw_effects) + self.ate
            
        elif effect_distribution == 'uniform':
            max_effect = kwargs.get('max_effect', 0.25)
            effects = np.random.uniform(-max_effect, max_effect, n)
            effects = effects - np.mean(effects) + self.ate
        
        # Count positive shifters
        positive_shifters = np.sum(effects > 0)
        proportion_positive = positive_shifters / n
        
        return {
            'positive_shifters': positive_shifters,
            'proportion_positive': proportion_positive,
            'negative_shifters': n - positive_shifters,
            'actual_ate': np.mean(effects),
            'target_ate': self.ate,
            'effects_distribution': effects
        }
    
    def policy_selection_score(self, baseline_support, ideological_score, 
                             plausibility_score, ceiling_risk=0.95):
        """
        Score a policy for endorsement experiment usefulness.
        
        Parameters:
        -----------
        baseline_support : float
            Proportion supporting policy (0-1)
        ideological_score : float
            Ideological neutrality (0=polarized, 1=neutral)
        plausibility_score : float
            Endorsement plausibility (0=implausible, 1=very plausible)
        ceiling_risk : float
            Threshold for ceiling effects
            
        Returns:
        --------
        dict: Policy scoring results
        """
        # Optimal baseline support is 0.8-0.9
        if baseline_support < 0.8:
            baseline_score = baseline_support / 0.8
        elif baseline_support <= 0.9:
            baseline_score = 1.0
        elif baseline_support < ceiling_risk:
            baseline_score = 1 - (baseline_support - 0.9) / (ceiling_risk - 0.9)
        else:
            baseline_score = 0.1  # High ceiling risk
            
        # Overall score (weighted average)
        overall_score = (0.4 * baseline_score + 
                        0.3 * ideological_score + 
                        0.3 * plausibility_score)
        
        return {
            'overall_score': overall_score,
            'baseline_score': baseline_score,
            'ideological_score': ideological_score,
            'plausibility_score': plausibility_score,
            'recommendation': 'Good' if overall_score > 0.7 else 'Moderate' if overall_score > 0.5 else 'Poor'
        }

def blair_replication():
    """
    Replicate the Blair et al. (2013) analysis with different interpretations.
    """
    print("=== BLAIR ET AL. (2013) REPLICATION ===\n")
    
    # Blair et al. parameters
    ate_blair = -0.011
    constant_blair = 0.8
    
    analyzer = EndorsementAnalyzer(ate_blair, constant_blair)
    
    # 1. Binary framework (ignoring baseline)
    print("1. BINARY FRAMEWORK (NO BASELINE INFO)")
    binary_results = analyzer.binary_framework_bounds()
    print(f"   Maximum supporters: {binary_results['p_max']:.1%}")
    print(f"   Interpretation: Up to {binary_results['p_max']:.1%} could support militant groups")
    print()
    
    # 2. Baseline support analysis
    print("2. BASELINE SUPPORT ANALYSIS")
    baseline_results = analyzer.baseline_support_analysis()
    print(f"   Control group average: {baseline_results['control_average']:.1%}")
    print(f"   Treatment group average: {baseline_results['treatment_average']:.1%}")
    print(f"   Proportion of supporters: {baseline_results['proportion_supporters']:.1%}")
    print(f"   {baseline_results['interpretation']}")
    print()
    
    # 3. Continuous effects bounds
    print("3. CONTINUOUS EFFECTS BOUNDS")
    continuous_results = analyzer.continuous_bounds()
    print(f"   Extreme scenario: {continuous_results['p_max_extreme']:.1%} supporters")
    print(f"   Symmetric effects: {continuous_results['p_symmetric']:.1%} supporters")
    if continuous_results['p_realistic']:
        print(f"   Realistic scenario: {continuous_results['p_realistic']:.1%} supporters")
    print()
    
    # 4. Simulation
    print("4. HETEROGENEOUS EFFECTS SIMULATION")
    sim_results = analyzer.simulate_heterogeneous_effects('normal', std=0.05)
    print(f"   Positive shifters: {sim_results['proportion_positive']:.1%}")
    print(f"   Target ATE: {sim_results['target_ate']:.4f}")
    print(f"   Simulated ATE: {sim_results['actual_ate']:.4f}")
    print()
    
    # 5. External validation
    print("5. EXTERNAL VALIDATION")
    print("   Pew Research (2015): 9-14% explicit support for TTP/JeM")
    print("   'Don't Know' responses: 30-49%")
    print("   Endorsement experiment suggests: 78.9% latent support")
    print("   → High 'DK' rates may reflect hidden preferences")
    print()
    
    return {
        'binary': binary_results,
        'baseline': baseline_results,
        'continuous': continuous_results,
        'simulation': sim_results
    }

def policy_selection_guide():
    """
    Demonstrate policy selection scoring for different policy types.
    """
    print("=== POLICY SELECTION GUIDE ===\n")
    
    analyzer = EndorsementAnalyzer(-0.011)  # ATE doesn't matter for scoring
    
    policies = [
        ("Infrastructure Development", 0.85, 0.9, 0.8),
        ("Polio Vaccination", 0.88, 0.95, 0.7),
        ("Military Spending", 0.65, 0.2, 0.6),
        ("Universal Healthcare", 0.70, 0.3, 0.5),
        ("Road Construction", 0.82, 0.85, 0.9),
        ("Gun Control", 0.45, 0.1, 0.2)
    ]
    
    print("Policy Scoring Results:")
    print("-" * 80)
    print(f"{'Policy':<25} {'Baseline':<10} {'Ideology':<10} {'Plausible':<10} {'Overall':<10} {'Rec':<10}")
    print("-" * 80)
    
    for policy_name, baseline, ideology, plausible in policies:
        scores = analyzer.policy_selection_score(baseline, ideology, plausible)
        print(f"{policy_name:<25} {baseline:<10.2f} {ideology:<10.2f} {plausible:<10.2f} " +
              f"{scores['overall_score']:<10.2f} {scores['recommendation']:<10}")
    
    print("-" * 80)
    print("\nScoring Criteria:")
    print("- Baseline: Optimal range 0.8-0.9 (high support, room for movement)")
    print("- Ideology: 1.0 = completely neutral, 0.0 = highly polarized")  
    print("- Plausible: 1.0 = very believable endorsement, 0.0 = implausible")
    print("- Overall: Weighted average (40% baseline, 30% ideology, 30% plausible)")

def create_visualizations(results):
    """
    Create visualizations for the analysis results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Endorsement Experiment Analysis: Blair et al. (2013)', fontsize=16, fontweight='bold')
    
    # 1. Comparison of different approaches
    approaches = ['Binary\n(No Baseline)', 'Baseline\nAnalysis', 'Continuous\n(Extreme)', 'Simulation']
    proportions = [
        results['binary']['p_max'],
        results['baseline']['proportion_supporters'],
        results['continuous']['p_max_extreme'],
        results['simulation']['proportion_positive']
    ]
    
    bars = axes[0,0].bar(approaches, proportions, color=['skyblue', 'orange', 'lightgreen', 'coral'])
    axes[0,0].set_ylabel('Proportion Supporting Group')
    axes[0,0].set_title('Support Estimates by Method')
    axes[0,0].set_ylim(0, 1)
    
    # Add percentage labels on bars
    for bar, prop in zip(bars, proportions):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                      f'{prop:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Distribution of simulated effects
    effects = results['simulation']['effects_distribution']
    axes[0,1].hist(effects, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0,1].axvline(0, color='red', linestyle='--', linewidth=2, label='No Effect')
    axes[0,1].axvline(np.mean(effects), color='orange', linestyle='-', linewidth=2, 
                     label=f'Mean ATE = {np.mean(effects):.4f}')
    axes[0,1].set_xlabel('Individual Treatment Effect')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Simulated Individual Effects Distribution')
    axes[0,1].legend()
    
    # 3. Policy selection heatmap
    policy_data = {
        'Infrastructure': [0.85, 0.9, 0.8],
        'Vaccination': [0.88, 0.95, 0.7],
        'Military': [0.65, 0.2, 0.6],
        'Healthcare': [0.70, 0.3, 0.5],
        'Roads': [0.82, 0.85, 0.9],
        'Gun Control': [0.45, 0.1, 0.2]
    }
    
    policy_df = pd.DataFrame(policy_data, index=['Baseline Support', 'Ideology Neutral', 'Plausible']).T
    sns.heatmap(policy_df, annot=True, cmap='RdYlGn', ax=axes[1,0], 
                vmin=0, vmax=1, cbar_kws={'label': 'Score (0-1)'})
    axes[1,0].set_title('Policy Selection Criteria Heatmap')
    axes[1,0].set_ylabel('Policy Type')
    
    # 4. Bounds comparison
    bound_types = ['Binary\nMax', 'Continuous\nExtreme', 'Continuous\nSymmetric', 'Baseline\nExact']
    bound_values = [
        results['binary']['p_max'],
        results['continuous']['p_max_extreme'],
        results['continuous']['p_symmetric'],
        results['baseline']['proportion_supporters']
    ]
    
    colors = ['lightblue', 'lightgreen', 'yellow', 'orange']
    bars = axes[1,1].bar(bound_types, bound_values, color=colors)
    axes[1,1].set_ylabel('Proportion Supporting Group')
    axes[1,1].set_title('Theoretical Bounds Comparison')
    axes[1,1].set_ylim(0, 1)
    
    # Add percentage labels
    for bar, val in zip(bars, bound_values):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{val:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Run main analysis
    print("ENDORSEMENT EXPERIMENT ANALYSIS")
    print("=" * 50)
    print()
    
    # Blair et al. replication
    results = blair_replication()
    
    # Policy selection guide
    policy_selection_guide()
    
    # Create visualizations
    fig = create_visualizations(results)
    plt.show()
    
    # Summary insights
    print("\n" + "=" * 50)
    print("KEY INSIGHTS")
    print("=" * 50)
    print("1. Baseline support information dramatically changes interpretation")
    print("   - Without baseline: Max 49.5% supporters")
    print("   - With baseline: 78.9% supporters")
    print()
    print("2. Small negative ATEs can mask majority support when baseline is high")
    print()
    print("3. Policy selection is crucial - need high baseline, neutral ideology")
    print()
    print("4. Computational approaches useful only for continuous heterogeneous effects")