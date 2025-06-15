"""
Utility functions for endorsement experiment analysis.

This module provides helper functions for the endorsement experiment framework.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy import stats
import warnings

def calculate_binary_bounds(ate):
    """
    Calculate theoretical bounds under binary switching framework.
    
    Parameters:
    -----------
    ate : float
        Average treatment effect
        
    Returns:
    --------
    tuple: (p_max, p_min) bounds on supporter proportion
    """
    p_max = (ate + 1) / 2
    p_min = max(0, (ate + 1) / 2)
    return p_max, p_min

def exact_support_from_baseline(ate, constant_term):
    """
    Calculate exact support proportion using baseline information.
    
    Under complete binary switching: treatment_avg = proportion_supporters
    
    Parameters:
    -----------
    ate : float
        Average treatment effect
    constant_term : float
        Baseline support level (control group average)
        
    Returns:
    --------
    dict: Analysis results
    """
    treatment_avg = constant_term + ate
    control_avg = constant_term
    
    return {
        'proportion_supporters': treatment_avg,
        'treatment_average': treatment_avg,
        'control_average': control_avg,
        'ate': ate,
        'interpretation': f"Exactly {treatment_avg:.1%} support the endorsing group"
    }

def continuous_effect_bounds(ate, max_individual_effect=1.0):
    """
    Calculate bounds under continuous effects with various assumptions.
    
    Parameters:
    -----------
    ate : float
        Average treatment effect
    max_individual_effect : float
        Maximum individual effect size
        
    Returns:
    --------
    dict: Various bound scenarios
    """
    # Scenario 1: Extreme opponent reaction (supporters unaffected)
    # ATE = p*0 + (1-p)*(-1) = -(1-p)
    p_max_extreme = 1 + ate
    
    # Scenario 2: Symmetric effects with maximum magnitude
    # ATE = p*(+max_effect) + (1-p)*(-max_effect) = max_effect*(2p-1)
    if max_individual_effect != 0:
        p_symmetric_max = ate / (2 * max_individual_effect) + 0.5
    else:
        p_symmetric_max = 0.5
    
    # Scenario 3: Small symmetric effects
    small_effect = max_individual_effect / 4
    if small_effect != 0:
        p_symmetric_small = ate / (2 * small_effect) + 0.5
    else:
        p_symmetric_small = 0.5
    
    return {
        'p_max_extreme': max(0, min(1, p_max_extreme)),
        'p_symmetric_max_effect': max(0, min(1, p_symmetric_max)),
        'p_symmetric_small_effect': max(0, min(1, p_symmetric_small)),
        'max_effect_used': max_individual_effect,
        'small_effect_used': small_effect
    }

def hungarian_optimization(control_responses, treatment_responses, maximize_positive=True):
    """
    Use Hungarian algorithm to find optimal pairing that maximizes positive shifters.
    
    Parameters:
    -----------
    control_responses : array-like
        Control group responses
    treatment_responses : array-like  
        Treatment group responses
    maximize_positive : bool
        Whether to maximize positive shifts
        
    Returns:
    --------
    dict: Optimization results
    """
    control_responses = np.array(control_responses)
    treatment_responses = np.array(treatment_responses)
    
    n_control = len(control_responses)
    n_treatment = len(treatment_responses)
    
    if n_control != n_treatment:
        min_size = min(n_control, n_treatment)
        control_responses = control_responses[:min_size]
        treatment_responses = treatment_responses[:min_size]
        warnings.warn(f"Sample sizes differ. Using first {min_size} observations from each group.")
    
    # Create cost matrix
    # cost[i,j] = cost of pairing control[i] with treatment[j]
    cost_matrix = np.zeros((len(control_responses), len(treatment_responses)))
    
    for i, control_val in enumerate(control_responses):
        for j, treatment_val in enumerate(treatment_responses):
            effect = treatment_val - control_val
            if maximize_positive:
                # Cost = -1 if positive effect (we want to maximize these)
                cost_matrix[i, j] = -1 if effect > 0 else 0
            else:
                # Cost = effect size (to minimize total effect)
                cost_matrix[i, j] = abs(effect)
    
    # Solve assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Calculate results
    effects = treatment_responses[col_indices] - control_responses[row_indices]
    positive_effects = np.sum(effects > 0)
    
    return {
        'positive_shifters': positive_effects,
        'total_pairs': len(effects),
        'proportion_positive': positive_effects / len(effects),
        'mean_effect': np.mean(effects),
        'effects': effects,
        'optimal_pairing': list(zip(row_indices, col_indices))
    }

def generate_realistic_responses(n, baseline_support=0.8, effect_std=0.05, 
                               response_distribution='beta'):
    """
    Generate realistic survey responses for simulation.
    
    Parameters:
    -----------
    n : int
        Sample size
    baseline_support : float
        Average support level
    effect_std : float
        Standard deviation of effects
    response_distribution : str
        'beta', 'normal', or 'uniform'
        
    Returns:
    --------
    dict: Generated control and treatment responses
    """
    if response_distribution == 'beta':
        # Beta distribution parameters to achieve desired mean
        # Mean of Beta(a,b) = a/(a+b)
        # Solve for parameters given desired mean and concentration
        concentration = 20  # Higher = less variance
        alpha = baseline_support * concentration
        beta = (1 - baseline_support) * concentration
        
        control = np.random.beta(alpha, beta, n)
        
    elif response_distribution == 'normal':
        control = np.random.normal(baseline_support, 0.1, n)
        control = np.clip(control, 0, 1)  # Bound to [0,1]
        
    elif response_distribution == 'uniform':
        # Uniform around baseline
        width = 0.3
        low = max(0, baseline_support - width/2)
        high = min(1, baseline_support + width/2)
        control = np.random.uniform(low, high, n)
    
    else:
        raise ValueError("response_distribution must be 'beta', 'normal', or 'uniform'")
    
    # Generate individual treatment effects
    individual_effects = np.random.normal(0, effect_std, n)
    
    # Create treatment responses
    treatment = control + individual_effects
    treatment = np.clip(treatment, 0, 1)  # Bound to [0,1]
    
    # Adjust to match target ATE if specified
    actual_ate = np.mean(treatment) - np.mean(control)
    
    return {
        'control': control,
        'treatment': treatment,
        'individual_effects': individual_effects,
        'actual_ate': actual_ate,
        'target_baseline': baseline_support
    }

def validate_experiment_power(ate, baseline_support, effect_size, alpha=0.05, power=0.8):
    """
    Calculate required sample size for detecting treatment effect.
    
    Parameters:
    -----------
    ate : float
        Expected average treatment effect
    baseline_support : float
        Baseline support level
    effect_size : float
        Expected individual effect size
    alpha : float
        Significance level
    power : float
        Desired statistical power
        
    Returns:
    --------
    dict: Power analysis results
    """
    # Estimate variance (rough approximation)
    # For binary outcomes: var ≈ p(1-p)
    # For continuous: assume variance around effect_size
    var_estimate = baseline_support * (1 - baseline_support) + effect_size**2
    
    # Effect size for power calculation (Cohen's d)
    cohens_d = abs(ate) / np.sqrt(var_estimate)
    
    # Rough sample size calculation (per group)
    # n ≈ 2 * (z_α/2 + z_β)² / d²
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    n_per_group = 2 * (z_alpha + z_beta)**2 / cohens_d**2
    
    return {
        'n_per_group': int(np.ceil(n_per_group)),
        'total_n': int(np.ceil(2 * n_per_group)),
        'cohens_d': cohens_d,
        'effect_size': 'Large' if cohens_d > 0.8 else 'Medium' if cohens_d > 0.5 else 'Small',
        'feasible': n_per_group < 5000  # Practical threshold
    }

def sensitivity_analysis(ate_range, constant_term, method='baseline'):
    """
    Perform sensitivity analysis across range of ATE values.
    
    Parameters:
    -----------
    ate_range : array-like
        Range of ATE values to test
    constant_term : float
        Baseline support level
    method : str
        'baseline', 'binary', or 'continuous'
        
    Returns:
    --------
    pandas.DataFrame: Sensitivity analysis results
    """
    results = []
    
    for ate in ate_range:
        if method == 'baseline':
            prop_supporters = constant_term + ate
        elif method == 'binary':
            prop_supporters = (ate + 1) / 2
        elif method == 'continuous':
            prop_supporters = 1 + ate  # Extreme scenario
        else:
            raise ValueError("Method must be 'baseline', 'binary', or 'continuous'")
        
        results.append({
            'ate': ate,
            'proportion_supporters': max(0, min(1, prop_supporters)),
            'method': method
        })
    
    return pd.DataFrame(results)

def compare_methods(ate, constant_term=None):
    """
    Compare all methods for estimating support proportion.
    
    Parameters:
    -----------
    ate : float
        Average treatment effect
    constant_term : float, optional
        Baseline support level
        
    Returns:
    --------
    pandas.DataFrame: Comparison of methods
    """
    results = []
    
    # Binary bounds
    p_max_binary, p_min_binary = calculate_binary_bounds(ate)
    results.append({
        'method': 'Binary (No Baseline)',
        'estimate': p_max_binary,
        'lower_bound': p_min_binary,
        'upper_bound': p_max_binary,
        'requires_baseline': False
    })
    
    # Baseline analysis
    if constant_term is not None:
        baseline_result = exact_support_from_baseline(ate, constant_term)
        results.append({
            'method': 'Baseline Analysis',
            'estimate': baseline_result['proportion_supporters'],
            'lower_bound': baseline_result['proportion_supporters'],
            'upper_bound': baseline_result['proportion_supporters'],
            'requires_baseline': True
        })
    
    # Continuous bounds
    continuous_result = continuous_effect_bounds(ate)
    results.append({
        'method': 'Continuous (Extreme)',
        'estimate': continuous_result['p_max_extreme'],
        'lower_bound': 0,
        'upper_bound': continuous_result['p_max_extreme'],
        'requires_baseline': False
    })
    
    results.append({
        'method': 'Continuous (Symmetric)',
        'estimate': continuous_result['p_symmetric_small_effect'],
        'lower_bound': continuous_result['p_symmetric_small_effect'],
        'upper_bound': continuous_result['p_symmetric_small_effect'],
        'requires_baseline': False
    })
    
    return pd.DataFrame(results)

# Example usage and testing
if __name__ == "__main__":
    print("Testing endorsement analysis utilities...")
    
    # Test Blair et al. parameters
    ate = -0.011
    constant = 0.8
    
    print(f"\nBlair et al. Analysis (ATE={ate}, Baseline={constant})")
    print("-" * 50)
    
    # Binary bounds
    p_max, p_min = calculate_binary_bounds(ate)
    print(f"Binary bounds: {p_min:.1%} - {p_max:.1%}")
    
    # Baseline analysis
    baseline_result = exact_support_from_baseline(ate, constant)
    print(f"Baseline analysis: {baseline_result['proportion_supporters']:.1%}")
    
    # Continuous bounds
    continuous_bounds = continuous_effect_bounds(ate)
    print(f"Continuous (extreme): {continuous_bounds['p_max_extreme']:.1%}")
    
    # Comparison table
    print("\nMethod Comparison:")
    comparison = compare_methods(ate, constant)
    print(comparison.round(3))
    
    print("\nUtilities loaded successfully!")