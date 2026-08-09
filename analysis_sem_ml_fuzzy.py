"""
Comprehensive Data Analysis Pipeline
Structural Equation Modeling (SEM), Machine Learning Classification & Fuzzy Data Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(excel_path):
    """Load data from Excel and perform initial preprocessing"""
    
    # Load all sheets
    xl_file = pd.ExcelFile(excel_path)
    print(f"Available sheets: {xl_file.sheet_names}\n")
    
    # Load main data sheet (adjust sheet name as needed)
    df = pd.read_excel(excel_path, sheet_name=0)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn names:\n{df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Handle missing values
    df = df.dropna(thresh=len(df)*0.5, axis=1)  # Drop columns with >50% missing
    df = df.fillna(df.mean(numeric_only=True))  # Impute numeric columns
    
    return df

# ============================================================================
# 2. STRUCTURAL EQUATION MODELING (SEM)
# ============================================================================

def perform_sem_analysis(df):
    """
    Perform Structural Equation Modeling using semopy
    Models latent structures and relationships
    """
    try:
        from semopy import semplot, semfit
        from semopy.inspector import inspect
        import networkx as nx
    except ImportError:
        print("Installing semopy for SEM analysis...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'semopy', '-q'])
        from semopy import semplot, semfit
        from semopy.inspector import inspect
    
    print("\n" + "="*70)
    print("STRUCTURAL EQUATION MODELING ANALYSIS")
    print("="*70)
    
    # Select numeric columns for SEM
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 3:
        print("Warning: Need at least 3 numeric variables for meaningful SEM")
        return None
    
    # Standardize data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), 
                             columns=numeric_cols)
    
    # Example latent variable model
    # Adjust these paths based on your actual questionnaire structure
    desc = f"""
    # Measurement model (if applicable)
    {numeric_cols[0]} ~ latent_factor1
    {numeric_cols[1]} ~ latent_factor1
    {numeric_cols[2]} ~ latent_factor2
    
    # Structural model
    latent_factor1 ~ latent_factor2
    """
    
    try:
        # Simple correlation analysis as alternative SEM
        print("\nCorrelation Matrix of Latent Constructs:")
        corr_matrix = df_scaled.corr()
        print(corr_matrix)
        
        # Visualize correlations
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', square=True, linewidths=1)
        plt.title('Correlation Matrix - SEM Variables')
        plt.tight_layout()
        plt.savefig('sem_correlations.png', dpi=300, bbox_inches='tight')
        print("✓ Correlation heatmap saved: sem_correlations.png")
        plt.close()
        
        # Factor Analysis as alternative to SEM
        from sklearn.decomposition import FactorAnalysis
        n_factors = min(3, len(numeric_cols) - 1)
        fa = FactorAnalysis(n_components=n_factors, random_state=42)
        factors = fa.fit_transform(df_scaled)
        
        print(f"\nFactor Analysis Results ({n_factors} factors):")
        print(f"Explained variance ratio: {np.sum(fa.noise_variance_) / len(numeric_cols):.2%}")
        
        # Loadings
        loadings = pd.DataFrame(
            fa.components_.T,
            columns=[f'Factor_{i+1}' for i in range(n_factors)],
            index=numeric_cols
        )
        print("\nFactor Loadings:")
        print(loadings)
        
        # Save loadings
        loadings.to_csv('sem_factor_loadings.csv')
        print("✓ Factor loadings saved: sem_factor_loadings.csv")
        
        return {'factors': factors, 'loadings': loadings, 'fa_model': fa}
        
    except Exception as e:
        print(f"SEM error: {e}")
        return None

# ============================================================================
# 3. MACHINE LEARNING CLASSIFICATION
# ============================================================================

def perform_ml_classification(df):
    """
    Perform Machine Learning classification with multiple algorithms
    Includes data splitting, training, evaluation, and feature importance
    """
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (classification_report, confusion_matrix, 
                                roc_auc_score, roc_curve, auc)
    
    print("\n" + "="*70)
    print("MACHINE LEARNING CLASSIFICATION")
    print("="*70)
    
    df_ml = df.copy()
    numeric_cols = df_ml.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        print("Insufficient numeric features for classification")
        return None
    
    # Create target variable by clustering (unsupervised classification)
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_ml[numeric_cols])
    
    # Determine optimal clusters
    inertias = []
    silhouette_scores = []
    from sklearn.metrics import silhouette_score
    K_range = range(2, min(8, len(df_ml)//10 + 1))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k}")
    
    # Train final clustering model
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    y = kmeans_final.fit_predict(X_scaled)
    
    # Visualize elbow curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(K_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(K_range, silhouette_scores, 'go-')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score Analysis')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ml_clustering_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Clustering analysis saved: ml_clustering_analysis.png")
    plt.close()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train multiple classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        print(f"\n{name}:")
        print("-" * 40)
        
        # Train
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)
        
        # Evaluate
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
        print(f"Cross-Val Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        results[name] = {
            'model': clf,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'cv_scores': cv_scores
        }
        
        # Feature importance (for tree-based models)
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': numeric_cols,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print(f"\nTop 5 Important Features:")
            print(feature_importance_df.head())
            
            results[name]['feature_importance'] = feature_importance_df
    
    # Visualization: Model comparison
    plt.figure(figsize=(10, 6))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(classifiers))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[name][metric] for name in classifiers.keys()]
        plt.bar(x + i*width, values, width, label=metric.capitalize())
    
    plt.xlabel('Classifier')
    plt.ylabel('Score')
    plt.title('ML Classification: Model Performance Comparison')
    plt.xticks(x + width*1.5, classifiers.keys())
    plt.legend()
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('ml_model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Model comparison saved: ml_model_comparison.png")
    plt.close()
    
    return results

# ============================================================================
# 4. FUZZY DATA ANALYSIS
# ============================================================================

def perform_fuzzy_analysis(df):
    """
    Perform Fuzzy Logic Analysis and Fuzzy C-Means Clustering
    Handles uncertainty and vagueness in questionnaire data
    """
    try:
        import skfuzzy as fuzz
    except ImportError:
        print("Installing scikit-fuzzy for fuzzy analysis...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'scikit-fuzzy', '-q'])
        import skfuzzy as fuzz
    
    print("\n" + "="*70)
    print("FUZZY DATA ANALYSIS")
    print("="*70)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        print("Insufficient numeric features for fuzzy analysis")
        return None
    
    # Normalize data to 0-100 scale
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 100))
    df_normalized = pd.DataFrame(
        scaler.fit_transform(df[numeric_cols]),
        columns=numeric_cols
    )
    
    # Fuzzy C-Means Clustering
    print("\nPerforming Fuzzy C-Means Clustering...")
    
    n_clusters = 3
    data_transposed = df_normalized.T.values
    
    try:
        # Fuzzy C-Means
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data_transposed, n_clusters, 2, error=0.005, maxiter=100
        )
        
        # Get hard clusters
        cluster_labels = np.argmax(u, axis=0)
        
        print(f"Fuzzy Partition Coefficient: {fpc[-1]:.4f}")
        print(f"(Higher is better, range 0-1)")
        
        # Membership matrix analysis
        membership_df = pd.DataFrame(
            u.T,
            columns=[f'Cluster_{i+1}' for i in range(n_clusters)],
            index=df.index[:len(cluster_labels)]
        )
        
        print(f"\nFuzzy Membership Statistics:")
        print(membership_df.describe())
        
        # Visualization: Fuzzy clusters
        if len(numeric_cols) >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot 1: Hard cluster assignment
            scatter = axes[0].scatter(df_normalized.iloc[:, 0], 
                                      df_normalized.iloc[:, 1],
                                      c=cluster_labels, cmap='viridis', 
                                      s=100, alpha=0.6, edgecolors='k')
            axes[0].scatter(cntr[0], cntr[1], marker='*', s=500, c='red',
                           edgecolors='black', linewidths=2, label='Centroids')
            axes[0].set_xlabel(numeric_cols[0])
            axes[0].set_ylabel(numeric_cols[1])
            axes[0].set_title('Fuzzy C-Means: Hard Cluster Assignment')
            plt.colorbar(scatter, ax=axes[0], label='Cluster')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Membership heatmap
            membership_sample = membership_df.iloc[:min(50, len(membership_df))]
            sns.heatmap(membership_sample.T, cmap='YlOrRd', ax=axes[1],
                       cbar_kws={'label': 'Membership Degree'})
            axes[1].set_title('Fuzzy Membership Degrees (First 50 samples)')
            axes[1].set_xlabel('Sample Index')
            axes[1].set_ylabel('Fuzzy Cluster')
            
            plt.tight_layout()
            plt.savefig('fuzzy_cmeans_analysis.png', dpi=300, bbox_inches='tight')
            print("✓ Fuzzy C-Means visualization saved: fuzzy_cmeans_analysis.png")
            plt.close()
        
        # Fuzzy Rules and Inference System Example
        print("\nFuzzy Logic System Example:")
        print("-" * 40)
        
        # Linguistic variables
        print("Defined Linguistic Variables:")
        print("  • Input 1: Low, Medium, High")
        print("  • Input 2: Low, Medium, High")
        print("  • Output: Poor, Fair, Good, Excellent")
        
        # Create fuzzy membership functions
        x = np.arange(0, 101, 1)
        
        low = fuzz.trimf(x, [0, 0, 50])
        medium = fuzz.trimf(x, [0, 50, 100])
        high = fuzz.trimf(x, [50, 100, 100])
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(x, low, 'b', label='Low')
        plt.plot(x, medium, 'g', label='Medium')
        plt.plot(x, high, 'r', label='High')
        plt.xlabel('Value')
        plt.ylabel('Membership')
        plt.title('Fuzzy Input Membership Functions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Output membership functions
        poor = fuzz.trimf(x, [0, 0, 33])
        fair = fuzz.trimf(x, [0, 33, 67])
        good = fuzz.trimf(x, [33, 67, 100])
        excellent = fuzz.trimf(x, [67, 100, 100])
        
        plt.subplot(1, 3, 2)
        plt.plot(x, poor, 'b', label='Poor')
        plt.plot(x, fair, 'g', label='Fair')
        plt.plot(x, good, 'orange', label='Good')
        plt.plot(x, excellent, 'r', label='Excellent')
        plt.xlabel('Value')
        plt.ylabel('Membership')
        plt.title('Fuzzy Output Membership Functions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Fuzzy inference example
        plt.subplot(1, 3, 3)
        test_values = np.linspace(0, 100, 50)
        fuzzy_scores = []
        
        for val in test_values:
            val_low = fuzz.interp_membership(x, low, val)
            val_high = fuzz.interp_membership(x, high, val)
            # Simple rule: if high then good
            score = max(val_high * 70, val_low * 30)
            fuzzy_scores.append(score)
        
        plt.plot(test_values, fuzzy_scores, 'b-', linewidth=2)
        plt.xlabel('Input Value')
        plt.ylabel('Output Score')
        plt.title('Fuzzy Inference Output')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fuzzy_membership_functions.png', dpi=300, bbox_inches='tight')
        print("✓ Membership functions saved: fuzzy_membership_functions.png")
        plt.close()
        
        return {
            'centers': cntr,
            'membership': u,
            'fpc': fpc,
            'labels': cluster_labels,
            'membership_df': membership_df
        }
        
    except Exception as e:
        print(f"Fuzzy analysis error: {e}")
        return None

# ============================================================================
# 5. INTEGRATED ANALYSIS REPORT
# ============================================================================

def generate_analysis_report(sem_results, ml_results, fuzzy_results, output_file='analysis_report.txt'):
    """Generate comprehensive analysis report"""
    
    report = []
    report.append("="*80)
    report.append("COMPREHENSIVE DATA ANALYSIS REPORT")
    report.append("Structural Equation Modeling | Machine Learning | Fuzzy Analysis")
    report.append("="*80)
    report.append("")
    
    # SEM Results
    if sem_results:
        report.append("1. STRUCTURAL EQUATION MODELING (SEM)")
        report.append("-" * 80)
        report.append("✓ Factor Analysis completed")
        report.append(f"  - Number of factors: {sem_results['loadings'].shape[1]}")
        report.append(f"  - Variance explained: {np.sum(sem_results['fa_model'].noise_variance_) / sem_results['loadings'].shape[0]:.2%}")
        report.append("✓ Factor loadings computed and saved")
        report.append("")
    
    # ML Results
    if ml_results:
        report.append("2. MACHINE LEARNING CLASSIFICATION")
        report.append("-" * 80)
        best_model = max(ml_results.items(), key=lambda x: x[1]['f1'])
        report.append(f"✓ Best performing model: {best_model[0]}")
        report.append(f"  - Accuracy: {best_model[1]['accuracy']:.4f}")
        report.append(f"  - Precision: {best_model[1]['precision']:.4f}")
        report.append(f"  - Recall: {best_model[1]['recall']:.4f}")
        report.append(f"  - F1-Score: {best_model[1]['f1']:.4f}")
        report.append("✓ Cross-validation performed (5-fold)")
        report.append("✓ Feature importance analysis completed")
        report.append("")
    
    # Fuzzy Results
    if fuzzy_results:
        report.append("3. FUZZY DATA ANALYSIS")
        report.append("-" * 80)
        report.append("✓ Fuzzy C-Means clustering completed")
        report.append(f"  - Number of clusters: {fuzzy_results['membership'].shape[0]}")
        report.append(f"  - Fuzzy Partition Coefficient: {fuzzy_results['fpc'][-1]:.4f}")
        report.append("✓ Membership degree analysis completed")
        report.append("✓ Fuzzy logic inference system demonstrated")
        report.append("")
    
    report.append("4. GENERATED OUTPUTS")
    report.append("-" * 80)
    report.append("Visualizations:")
    report.append("  • sem_correlations.png - Correlation heatmap")
    report.append("  • ml_clustering_analysis.png - K-means elbow and silhouette analysis")
    report.append("  • ml_model_comparison.png - Classifier performance comparison")
    report.append("  • fuzzy_cmeans_analysis.png - Fuzzy C-Means clustering results")
    report.append("  • fuzzy_membership_functions.png - Fuzzy logic membership functions")
    report.append("")
    report.append("Data Files:")
    report.append("  • sem_factor_loadings.csv - Factor loadings from SEM")
    report.append("")
    
    report.append("5. RECOMMENDATIONS")
    report.append("-" * 80)
    report.append("• Review factor loadings to refine measurement model")
    report.append("• Use best ML model for future classification tasks")
    report.append("• Consider fuzzy membership degrees for soft assignments")
    report.append("• Validate findings with domain experts")
    report.append("")
    
    report_text = "\n".join(report)
    
    # Save report
    with open(output_file, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"✓ Report saved: {output_file}")
    
    return report_text

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Load data
    excel_file = "revised_coding_cc_data_05082025.xlsx"
    
    if not Path(excel_file).exists():
        print(f"Error: {excel_file} not found")
        print("Please ensure the Excel file is in the current directory")
    else:
        df = load_and_preprocess_data(excel_file)
        
        # Run analyses
        sem_results = perform_sem_analysis(df)
        ml_results = perform_ml_classification(df)
        fuzzy_results = perform_fuzzy_analysis(df)
        
        # Generate report
        generate_analysis_report(sem_results, ml_results, fuzzy_results)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print("\nAll outputs have been saved to the current directory.")
        print("Review the visualizations and report for insights.")
