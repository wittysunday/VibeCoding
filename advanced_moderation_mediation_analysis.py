"""
Advanced Analysis Module: Moderation, Mediation & Visualization
Extended Statistical Modeling with Results Display and Diagnostics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ============================================================================
# 1. MODERATION ANALYSIS
# ============================================================================

def perform_moderation_analysis(df, predictor_col, outcome_col, moderator_col, 
                                output_prefix='moderation'):
    """
    Perform moderation analysis to test if moderator affects the relationship
    between predictor and outcome.
    
    Model: Y = b0 + b1*X + b2*M + b3*X*M + error
    
    Args:
        df: DataFrame with data
        predictor_col: Name of predictor variable (X)
        outcome_col: Name of outcome variable (Y)
        moderator_col: Name of moderator variable (M)
        output_prefix: Prefix for output files
    """
    
    print("\n" + "="*80)
    print("MODERATION ANALYSIS")
    print("="*80)
    print(f"Model: {outcome_col} ~ {predictor_col} * {moderator_col}")
    print("-"*80)
    
    # Prepare data
    data = df[[predictor_col, outcome_col, moderator_col]].dropna()
    
    # Standardize variables (improves interpretability and reduces multicollinearity)
    scaler = StandardScaler()
    X = data[[predictor_col, moderator_col]].values
    X_scaled = scaler.fit_transform(X)
    
    X_pred = X_scaled[:, 0:1]  # Predictor
    M_mod = X_scaled[:, 1:2]   # Moderator
    y = data[outcome_col].values
    
    # Create interaction term
    X_M = X_pred * M_mod
    
    # Fit regression: Y = b0 + b1*X + b2*M + b3*X*M
    X_model = np.column_stack([X_pred, M_mod, X_M])
    X_model = np.column_stack([np.ones(len(X_model)), X_model])  # Add intercept
    
    beta = np.linalg.lstsq(X_model, y, rcond=None)[0]
    y_pred = X_model @ beta
    
    # Calculate statistics
    residuals = y - y_pred
    n = len(y)
    k = X_model.shape[1] - 1  # Number of predictors (excluding intercept)
    
    r2 = r2_score(y, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    # Standard errors and t-statistics
    var_residuals = np.sum(residuals**2) / (n - k - 1)
    var_beta = var_residuals * np.linalg.inv(X_model.T @ X_model)
    se_beta = np.sqrt(np.diag(var_beta))
    t_stats = beta / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))
    
    # Results DataFrame
    results_df = pd.DataFrame({
        'Variable': ['Intercept', predictor_col, moderator_col, f'{predictor_col}*{moderator_col}'],
        'Coefficient': beta,
        'Std Error': se_beta,
        't-statistic': t_stats,
        'p-value': p_values,
        'Significant': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '' 
                       for p in p_values]
    })
    
    print("\nREGRESSION RESULTS:")
    print(results_df.to_string(index=False))
    print(f"\nModel Fit:")
    print(f"  R² = {r2:.4f}")
    print(f"  Adjusted R² = {adj_r2:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  N = {n}")
    
    # Interpretation
    b1 = beta[1]  # Direct effect of X
    b3 = beta[3]  # Moderation effect (X*M)
    
    print(f"\nINTERPRETATION:")
    print(f"  Direct effect (b1): {b1:.4f} (p={p_values[1]:.4f})")
    print(f"  Moderation effect (b3): {b3:.4f} (p={p_values[3]:.4f})")
    
    if p_values[3] < 0.05:
        print(f"  ✓ MODERATION SIGNIFICANT: The effect of {predictor_col} on {outcome_col}")
        print(f"    depends on {moderator_col}")
        if b3 > 0:
            print(f"    → Higher {moderator_col} STRENGTHENS the relationship")
        else:
            print(f"    → Higher {moderator_col} WEAKENS the relationship")
    else:
        print(f"  ✗ MODERATION NOT SIGNIFICANT: The relationship between {predictor_col}")
        print(f"    and {outcome_col} does not significantly depend on {moderator_col}")
    
    # Visualization: Simple slopes analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Interaction plot
    m_low = np.percentile(data[moderator_col], 25)
    m_mid = np.percentile(data[moderator_col], 50)
    m_high = np.percentile(data[moderator_col], 75)
    
    x_range = np.linspace(data[predictor_col].min(), data[predictor_col].max(), 100)
    
    # Scale for prediction
    x_scaled_range = (x_range - scaler.mean_[0]) / scaler.scale_[0]
    m_low_scaled = (m_low - scaler.mean_[1]) / scaler.scale_[1]
    m_mid_scaled = (m_mid - scaler.mean_[1]) / scaler.scale_[1]
    m_high_scaled = (m_high - scaler.mean_[1]) / scaler.scale_[1]
    
    y_low = beta[0] + beta[1]*x_scaled_range + beta[2]*m_low_scaled + beta[3]*x_scaled_range*m_low_scaled
    y_mid = beta[0] + beta[1]*x_scaled_range + beta[2]*m_mid_scaled + beta[3]*x_scaled_range*m_mid_scaled
    y_high = beta[0] + beta[1]*x_scaled_range + beta[2]*m_high_scaled + beta[3]*x_scaled_range*m_high_scaled
    
    axes[0].plot(x_range, y_low, 'b-', linewidth=2, label=f'{moderator_col} Low (25th %ile)')
    axes[0].plot(x_range, y_mid, 'g-', linewidth=2, label=f'{moderator_col} Medium (50th %ile)')
    axes[0].plot(x_range, y_high, 'r-', linewidth=2, label=f'{moderator_col} High (75th %ile)')
    axes[0].scatter(data[predictor_col], y, alpha=0.3, s=30, color='gray', edgecolors='k', linewidth=0.5)
    axes[0].set_xlabel(predictor_col, fontsize=12)
    axes[0].set_ylabel(outcome_col, fontsize=12)
    axes[0].set_title('Moderation Analysis: Simple Slopes', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals diagnostic
    axes[1].scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Fitted Values', fontsize=12)
    axes[1].set_ylabel('Residuals', fontsize=12)
    axes[1].set_title('Residual Plot', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_prefix}_analysis.png")
    plt.close()
    
    # Save results
    results_df.to_csv(f'{output_prefix}_results.csv', index=False)
    print(f"✓ Results saved: {output_prefix}_results.csv")
    
    return {
        'results': results_df,
        'r2': r2,
        'adj_r2': adj_r2,
        'rmse': rmse,
        'coefficients': beta,
        'p_values': p_values,
        'significant': p_values[3] < 0.05
    }

# ============================================================================
# 2. MEDIATION ANALYSIS
# ============================================================================

def perform_mediation_analysis(df, predictor_col, outcome_col, mediator_col,
                               output_prefix='mediation'):
    """
    Perform mediation analysis using Baron & Kenny approach with Sobel test.
    
    Models:
    Total effect: Y = c*X + error (c path)
    Direct effect: Y = c'*X + b*M + error (c' and b paths)
    Mediator: M = a*X + error (a path)
    
    Indirect effect = a*b
    Sobel test: z = a*b / sqrt(b²*SE_a² + a²*SE_b²)
    
    Args:
        df: DataFrame with data
        predictor_col: Name of predictor variable (X)
        outcome_col: Name of outcome variable (Y)
        mediator_col: Name of mediator variable (M)
        output_prefix: Prefix for output files
    """
    
    print("\n" + "="*80)
    print("MEDIATION ANALYSIS")
    print("="*80)
    print(f"Model: {outcome_col} ← {predictor_col} → {mediator_col} → {outcome_col}")
    print("-"*80)
    
    # Prepare data
    data = df[[predictor_col, outcome_col, mediator_col]].dropna()
    n = len(data)
    
    X = data[predictor_col].values.reshape(-1, 1)
    Y = data[outcome_col].values.reshape(-1, 1)
    M = data[mediator_col].values.reshape(-1, 1)
    
    # Add intercept
    X_intercept = np.column_stack([np.ones(n), X])
    
    # Step 1: Total effect (c path)
    # Y = c*X
    reg_total = np.linalg.lstsq(X_intercept, Y, rcond=None)[0]
    c_coeff = reg_total[1, 0]
    
    residuals_total = Y - X_intercept @ reg_total
    var_residuals_total = np.sum(residuals_total**2) / (n - 2)
    var_c = var_residuals_total * np.linalg.inv(X_intercept.T @ X_intercept)[1, 1]
    se_c = np.sqrt(var_c)
    t_c = c_coeff / se_c
    p_c = 2 * (1 - stats.t.cdf(np.abs(t_c), n - 2))
    
    r2_total = 1 - np.sum(residuals_total**2) / np.sum((Y - np.mean(Y))**2)
    
    # Step 2: Direct effect and a path
    # M = a*X
    reg_mediator = np.linalg.lstsq(X_intercept, M, rcond=None)[0]
    a_coeff = reg_mediator[1, 0]
    
    residuals_mediator = M - X_intercept @ reg_mediator
    var_residuals_mediator = np.sum(residuals_mediator**2) / (n - 2)
    var_a = var_residuals_mediator * np.linalg.inv(X_intercept.T @ X_intercept)[1, 1]
    se_a = np.sqrt(var_a)
    t_a = a_coeff / se_a
    p_a = 2 * (1 - stats.t.cdf(np.abs(t_a), n - 2))
    
    # Step 3: Direct effect (c' path) and indirect effect
    # Y = c'*X + b*M
    XM = np.column_stack([np.ones(n), X, M])
    reg_direct = np.linalg.lstsq(XM, Y, rcond=None)[0]
    c_prime = reg_direct[1, 0]
    b_coeff = reg_direct[2, 0]
    
    residuals_direct = Y - XM @ reg_direct
    var_residuals_direct = np.sum(residuals_direct**2) / (n - 3)
    var_matrix = var_residuals_direct * np.linalg.inv(XM.T @ XM)
    var_c_prime = var_matrix[1, 1]
    var_b = var_matrix[2, 2]
    se_c_prime = np.sqrt(var_c_prime)
    se_b = np.sqrt(var_b)
    t_c_prime = c_prime / se_c_prime
    t_b = b_coeff / se_b
    p_c_prime = 2 * (1 - stats.t.cdf(np.abs(t_c_prime), n - 3))
    p_b = 2 * (1 - stats.t.cdf(np.abs(t_b), n - 3))
    
    r2_direct = 1 - np.sum(residuals_direct**2) / np.sum((Y - np.mean(Y))**2)
    
    # Indirect effect and Sobel test
    indirect_effect = a_coeff * b_coeff
    se_indirect = np.sqrt(b_coeff**2 * se_a**2 + a_coeff**2 * se_b**2)
    z_sobel = indirect_effect / se_indirect if se_indirect > 0 else 0
    p_indirect = 2 * (1 - stats.norm.cdf(np.abs(z_sobel)))
    
    # Results display
    print("\nPATH COEFFICIENTS:")
    print("-"*80)
    print(f"Total Effect (c):     {c_coeff:8.4f}  SE={se_c:.4f}  t={t_c:7.3f}  p={p_c:.4f} {'***' if p_c < 0.001 else '**' if p_c < 0.01 else '*' if p_c < 0.05 else ''}")
    print(f"  → Effect of {predictor_col} on {outcome_col} (ignoring mediator)")
    print()
    print(f"Path a:               {a_coeff:8.4f}  SE={se_a:.4f}  t={t_a:7.3f}  p={p_a:.4f} {'***' if p_a < 0.001 else '**' if p_a < 0.01 else '*' if p_a < 0.05 else ''}")
    print(f"  → Effect of {predictor_col} on {mediator_col}")
    print()
    print(f"Path b:               {b_coeff:8.4f}  SE={se_b:.4f}  t={t_b:7.3f}  p={p_b:.4f} {'***' if p_b < 0.001 else '**' if p_b < 0.01 else '*' if p_b < 0.05 else ''}")
    print(f"  → Effect of {mediator_col} on {outcome_col}")
    print()
    print(f"Direct Effect (c'):   {c_prime:8.4f}  SE={se_c_prime:.4f}  t={t_c_prime:7.3f}  p={p_c_prime:.4f} {'***' if p_c_prime < 0.001 else '**' if p_c_prime < 0.01 else '*' if p_c_prime < 0.05 else ''}")
    print(f"  → Effect of {predictor_col} on {outcome_col} (controlling for mediator)")
    print()
    print(f"Indirect Effect:      {indirect_effect:8.4f}  SE={se_indirect:.4f}")
    print(f"Sobel Test:           z={z_sobel:8.4f}  p={p_indirect:.4f} {'***' if p_indirect < 0.001 else '**' if p_indirect < 0.01 else '*' if p_indirect < 0.05 else ''}")
    
    print(f"\nMODEL FIT:")
    print(f"  Total effect model R² = {r2_total:.4f}")
    print(f"  Direct effect model R² = {r2_direct:.4f}")
    print(f"  Mediation explains {(r2_direct - r2_total):.4f} additional variance")
    
    # Mediation type
    print(f"\nMEDIATION TYPE:")
    indirect_sig = p_indirect < 0.05
    direct_sig = p_c_prime < 0.05
    
    if indirect_sig and not direct_sig:
        print(f"  → FULL MEDIATION: Indirect effect is significant, direct effect is not")
        print(f"     {mediator_col} completely explains the relationship")
    elif indirect_sig and direct_sig:
        print(f"  → PARTIAL MEDIATION: Both indirect and direct effects are significant")
        print(f"     {mediator_col} partially explains the relationship")
    elif not indirect_sig:
        print(f"  → NO MEDIATION: Indirect effect is not significant")
        print(f"     {mediator_col} does not mediate the relationship")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Path diagram (conceptual)
    ax = axes[0, 0]
    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 10)
    ax.axis('off')
    
    # Draw boxes for variables
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    
    box_predictor = FancyBboxPatch((0.5, 4), 2, 1.5, boxstyle="round,pad=0.1", 
                                   edgecolor='blue', facecolor='lightblue', linewidth=2)
    box_mediator = FancyBboxPatch((4, 4), 2, 1.5, boxstyle="round,pad=0.1",
                                  edgecolor='green', facecolor='lightgreen', linewidth=2)
    box_outcome = FancyBboxPatch((7.5, 4), 2, 1.5, boxstyle="round,pad=0.1",
                                 edgecolor='red', facecolor='lightcoral', linewidth=2)
    
    ax.add_patch(box_predictor)
    ax.add_patch(box_mediator)
    ax.add_patch(box_outcome)
    
    ax.text(1.5, 4.75, predictor_col, ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5, 4.75, mediator_col, ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(8.5, 4.75, outcome_col, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw arrows
    arrow_a = FancyArrowPatch((2.5, 4.75), (4, 4.75), arrowstyle='->', mutation_scale=20, 
                              linewidth=2, color='green')
    arrow_b = FancyArrowPatch((6, 4.75), (7.5, 4.75), arrowstyle='->', mutation_scale=20,
                              linewidth=2, color='green')
    arrow_c_prime = FancyArrowPatch((2.5, 4.3), (7.5, 4.3), arrowstyle='->', mutation_scale=20,
                                    linewidth=2, color='red', linestyle='--')
    arrow_c = FancyArrowPatch((2.5, 5.2), (7.5, 5.2), arrowstyle='->', mutation_scale=20,
                              linewidth=2, color='blue', linestyle=':')
    
    ax.add_patch(arrow_a)
    ax.add_patch(arrow_b)
    ax.add_patch(arrow_c_prime)
    ax.add_patch(arrow_c)
    
    ax.text(3.25, 5.1, f'a={a_coeff:.3f}', ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax.text(6.25, 5.1, f'b={b_coeff:.3f}', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax.text(5, 3.5, f"c'={c_prime:.3f}", ha='center', fontsize=10, color='red',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax.text(5, 6, f'c={c_coeff:.3f}', ha='center', fontsize=10, color='blue',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    ax.set_title('Mediation Path Diagram', fontsize=13, fontweight='bold')
    
    # Plot 2: Effect sizes
    ax = axes[0, 1]
    effects = [c_coeff, a_coeff, b_coeff, c_prime, indirect_effect]
    effect_names = ['Total\nEffect (c)', f'Path a\n({predictor_col}→{mediator_col})', 
                    f'Path b\n({mediator_col}→{outcome_col})', 
                    f'Direct\nEffect (c\')', 'Indirect\nEffect (a×b)']
    colors = ['blue' if e > 0 else 'red' for e in effects]
    
    bars = ax.bar(effect_names, effects, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Coefficient Value', fontsize=11)
    ax.set_title('Effect Sizes Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # Plot 3: X → Y relationship
    ax = axes[1, 0]
    ax.scatter(data[predictor_col], data[outcome_col], alpha=0.5, s=50, edgecolors='k', linewidth=0.5)
    z = np.polyfit(data[predictor_col], data[outcome_col], 1)
    p = np.poly1d(z)
    x_line = np.linspace(data[predictor_col].min(), data[predictor_col].max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Total effect (c={c_coeff:.3f})')
    ax.set_xlabel(predictor_col, fontsize=11)
    ax.set_ylabel(outcome_col, fontsize=11)
    ax.set_title('Total Effect (X → Y)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: X → M → Y pathway
    ax = axes[1, 1]
    Y_predicted = reg_direct[0] + c_prime * X + b_coeff * M
    residuals_plot = Y - Y_predicted
    ax.scatter(Y_predicted.flatten(), residuals_plot.flatten(), alpha=0.5, s=50, edgecolors='k', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Fitted Values', fontsize=11)
    ax.set_ylabel('Residuals', fontsize=11)
    ax.set_title('Mediation Model: Residual Plot', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_prefix}_analysis.png")
    plt.close()
    
    # Save results
    results_df = pd.DataFrame({
        'Path': ['Total Effect (c)', 'Path a', 'Path b', 'Direct Effect (c\')', 'Indirect Effect', 'Sobel Test'],
        'Coefficient': [c_coeff, a_coeff, b_coeff, c_prime, indirect_effect, z_sobel],
        'Std Error': [se_c, se_a, se_b, se_c_prime, se_indirect, np.nan],
        'p-value': [p_c, p_a, p_b, p_c_prime, p_indirect, p_indirect],
        'Significant': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '' 
                       for p in [p_c, p_a, p_b, p_c_prime, p_indirect, p_indirect]]
    })
    
    results_df.to_csv(f'{output_prefix}_results.csv', index=False)
    print(f"✓ Results saved: {output_prefix}_results.csv")
    
    return {
        'results': results_df,
        'paths': {
            'c': c_coeff,
            'a': a_coeff,
            'b': b_coeff,
            'c_prime': c_prime,
            'indirect': indirect_effect
        },
        'p_values': {
            'c': p_c,
            'a': p_a,
            'b': p_b,
            'c_prime': p_c_prime,
            'indirect': p_indirect
        },
        'mediation_type': 'Full' if (indirect_sig and not direct_sig) else 
                         'Partial' if (indirect_sig and direct_sig) else 'None'
    }

# ============================================================================
# 3. COMPREHENSIVE VISUALIZATION SUITE
# ============================================================================

def create_comprehensive_visualizations(df, output_prefix='comprehensive'):
    """
    Create a comprehensive set of visualizations for data exploration
    """
    
    print("\n" + "="*80)
    print("CREATING COMPREHENSIVE VISUALIZATIONS")
    print("="*80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        print("Insufficient numeric columns for visualization")
        return
    
    # Figure 1: Distribution analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Data Distribution Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    # Histogram
    axes[0, 0].hist(df[numeric_cols[0]], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel(numeric_cols[0], fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Distribution (Histogram)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Q-Q plot for normality
    stats.probplot(df[numeric_cols[0]].dropna(), dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plot
    axes[1, 0].boxplot([df[col].dropna() for col in numeric_cols[:min(4, len(numeric_cols))]],
                       labels=numeric_cols[:min(4, len(numeric_cols))])
    axes[1, 0].set_ylabel('Value', fontsize=11)
    axes[1, 0].set_title('Box Plot Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Density plot
    for col in numeric_cols[:min(3, len(numeric_cols))]:
        df[col].plot(kind='density', ax=axes[1, 1], label=col, linewidth=2)
    axes[1, 1].set_title('Density Plot', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Value', fontsize=11)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_distributions.png', dpi=300, bbox_inches='tight')
    print("✓ Distribution visualization saved")
    plt.close()
    
    # Figure 2: Correlation matrix with hierarchical clustering
    fig, ax = plt.subplots(figsize=(12, 10))
    
    corr_matrix = df[numeric_cols].corr()
    
    # Hierarchical clustering on correlation matrix
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    
    # Convert correlation to distance
    distance_matrix = 1 - corr_matrix.abs()
    condensed_dist = squareform(distance_matrix)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='ward')
    dendro = dendrogram(linkage_matrix, labels=corr_matrix.columns, no_plot=True)
    order = dendro['leaves']
    
    # Reorder correlation matrix
    corr_reordered = corr_matrix.iloc[order, order]
    
    # Plot heatmap
    sns.heatmap(corr_reordered, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={'label': 'Correlation'},
                ax=ax, vmin=-1, vmax=1)
    ax.set_title('Correlation Matrix (Hierarchical Clustering)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Correlation matrix visualization saved")
    plt.close()
    
    # Figure 3: Scatter matrix with regression lines
    n_vars = min(4, len(numeric_cols))
    vars_to_plot = numeric_cols[:n_vars]
    
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(14, 14))
    fig.suptitle('Scatter Plot Matrix with Regression Lines', fontsize=16, fontweight='bold')
    
    for i, var_y in enumerate(vars_to_plot):
        for j, var_x in enumerate(vars_to_plot):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: distribution
                ax.hist(df[var_x].dropna(), bins=20, color='skyblue', edgecolor='black', alpha=0.7)
                ax.set_ylabel('Frequency', fontsize=9)
            else:
                # Off-diagonal: scatter with regression line
                valid_data = df[[var_x, var_y]].dropna()
                ax.scatter(valid_data[var_x], valid_data[var_y], alpha=0.5, s=30, edgecolors='k', linewidth=0.3)
                
                # Regression line
                z = np.polyfit(valid_data[var_x], valid_data[var_y], 1)
                p = np.poly1d(z)
                x_line = np.linspace(valid_data[var_x].min(), valid_data[var_x].max(), 100)
                ax.plot(x_line, p(x_line), 'r-', linewidth=2, alpha=0.8)
                
                # Correlation coefficient
                r = np.corrcoef(valid_data[var_x], valid_data[var_y])[0, 1]
                ax.text(0.05, 0.95, f'r={r:.2f}', transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
            
            if j == 0:
                ax.set_ylabel(var_y, fontsize=10)
            if i == n_vars - 1:
                ax.set_xlabel(var_x, fontsize=10)
            
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_scatter_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Scatter matrix visualization saved")
    plt.close()
    
    print("✓ All visualizations created successfully\n")

# ============================================================================
# 4. RESULTS SUMMARY REPORT
# ============================================================================

def generate_advanced_report(moderation_results, mediation_results, output_file='advanced_analysis_report.txt'):
    """Generate comprehensive report with all analyses"""
    
    report = []
    report.append("="*80)
    report.append("ADVANCED STATISTICAL ANALYSIS REPORT")
    report.append("Moderation & Mediation Analyses with Visualizations")
    report.append("="*80)
    report.append("")
    
    if moderation_results:
        report.append("1. MODERATION ANALYSIS RESULTS")
        report.append("-"*80)
        report.append(f"R² = {moderation_results['r2']:.4f}")
        report.append(f"Adjusted R² = {moderation_results['adj_r2']:.4f}")
        report.append(f"RMSE = {moderation_results['rmse']:.4f}")
        report.append("")
        report.append(f"Moderation Effect Significant: {moderation_results['significant']}")
        report.append(f"Interpretation: {'The moderator significantly affects the relationship' if moderation_results['significant'] else 'No significant moderation effect detected'}")
        report.append("")
    
    if mediation_results:
        report.append("2. MEDIATION ANALYSIS RESULTS")
        report.append("-"*80)
        report.append(f"Mediation Type: {mediation_results['mediation_type']} Mediation")
        report.append(f"Indirect Effect: {mediation_results['paths']['indirect']:.4f} (p={mediation_results['p_values']['indirect']:.4f})")
        report.append(f"Direct Effect: {mediation_results['paths']['c_prime']:.4f} (p={mediation_results['p_values']['c_prime']:.4f})")
        report.append("")
        report.append(f"Interpretation: ", end='')
        if mediation_results['mediation_type'] == 'Full':
            report.append("The mediator completely explains the relationship")
        elif mediation_results['mediation_type'] == 'Partial':
            report.append("The mediator partially explains the relationship")
        else:
            report.append("No mediation effect detected")
        report.append("")
    
    report.append("3. OUTPUT FILES GENERATED")
    report.append("-"*80)
    report.append("Moderation Analysis:")
    report.append("  • moderation_analysis.png - Simple slopes visualization")
    report.append("  • moderation_results.csv - Detailed regression statistics")
    report.append("")
    report.append("Mediation Analysis:")
    report.append("  • mediation_analysis.png - Path diagram and effect visualizations")
    report.append("  • mediation_results.csv - Detailed path analysis results")
    report.append("")
    report.append("Comprehensive Visualizations:")
    report.append("  • comprehensive_distributions.png - Histogram, Q-Q plot, boxplots")
    report.append("  • comprehensive_correlation_matrix.png - Clustered correlations")
    report.append("  • comprehensive_scatter_matrix.png - Regression scatter plots")
    report.append("")
    
    report_text = "\n".join(report)
    
    with open(output_file, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"✓ Report saved: {output_file}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Load data from earlier analysis
    excel_file = "revised_coding_cc_data_05082025.xlsx"
    
    if not Path(excel_file).exists():
        print(f"Error: {excel_file} not found")
    else:
        print("Loading data...")
        df = pd.read_excel(excel_file)
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 3:
            # Perform moderation analysis
            print("\nPerforming moderation analysis...")
            mod_results = perform_moderation_analysis(
                df, 
                predictor_col=numeric_cols[0],
                outcome_col=numeric_cols[1],
                moderator_col=numeric_cols[2],
                output_prefix='moderation'
            )
            
            # Perform mediation analysis
            print("\nPerforming mediation analysis...")
            med_results = perform_mediation_analysis(
                df,
                predictor_col=numeric_cols[0],
                outcome_col=numeric_cols[1],
                mediator_col=numeric_cols[2],
                output_prefix='mediation'
            )
            
        # Create comprehensive visualizations
        print("\nCreating comprehensive visualizations...")
        create_comprehensive_visualizations(df, output_prefix='comprehensive')
        
        # Generate report
        print("\nGenerating advanced analysis report...")
        if len(numeric_cols) >= 3:
            generate_advanced_report(mod_results, med_results)
        
        print("\n" + "="*80)
        print("ADVANCED ANALYSIS COMPLETE")
        print("="*80)
        print("\nAll results, visualizations, and reports have been generated.")
        print("Check the output files for detailed interpretations.")
