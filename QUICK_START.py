"""
QUICK START GUIDE - Step-by-Step Tutorial
VibeCoding Analysis Pipeline
"""

# ============================================================================
# STEP 1: INSTALLATION
# ============================================================================

"""
Run these commands in your terminal:

1. Navigate to project directory:
   cd VibeCoding

2. Create virtual environment:
   python -m venv venv
   
3. Activate virtual environment:
   On Windows:  venv\Scripts\activate
   On Mac/Linux: source venv/bin/activate

4. Install dependencies:
   pip install -r requirements.txt

That's it! You're ready to run the analysis.
"""

# ============================================================================
# STEP 2: PREPARE YOUR DATA
# ============================================================================

"""
Your Excel file should have:
  • Column 1: Respondent ID (optional)
  • Columns 2+: Questionnaire responses (numeric values)
  • Rows: Individual respondent entries

Expected format for revised_coding_cc_data_05082025.xlsx:
┌─────┬──────────┬──────────┬──────────┐
│ ID  │ Question1│ Question2│ Question3│
├─────┼──────────┼──────────┼──────────┤
│ 1   │ 4        │ 5        │ 3        │
│ 2   │ 5        │ 4        │ 4        │
│ 3   │ 3        │ 3        │ 2        │
└─────┴──────────┴──────────┴──────────┘

Numeric values: Can be Likert scale (1-5), percentages (0-100), 
or any numeric format. Script will auto-normalize.
"""

# ============================================================================
# STEP 3: RUN THE ANALYSIS
# ============================================================================

"""
Simple execution:

  1. Open terminal/command prompt
  2. Navigate to VibeCoding directory
  3. Run: python analysis_sem_ml_fuzzy.py
  4. Wait for completion (2-5 minutes depending on data size)
  
The script will:
  ✓ Load your Excel data
  ✓ Preprocess and clean data
  ✓ Run Structural Equation Modeling
  ✓ Train Machine Learning classifiers
  ✓ Perform Fuzzy analysis
  ✓ Generate visualizations
  ✓ Create summary report
"""

# ============================================================================
# STEP 4: INTERPRET RESULTS
# ============================================================================

"""
CHECK THESE OUTPUT FILES IN ORDER:

1. ANALYSIS_REPORT.TXT
   └─ High-level summary of findings
   └─ Which model performed best
   └─ Key statistics and metrics

2. SEM_CORRELATIONS.PNG
   └─ Look for patterns in variable relationships
   └─ Strong correlations (dark red) show related constructs
   └─ Weak correlations (light blue) show independent dimensions

3. ML_CLUSTERING_ANALYSIS.PNG
   ├─ LEFT PLOT (Elbow): Where does curve "bend"?
   │  └─ Suggests optimal number of distinct groups
   ├─ RIGHT PLOT (Silhouette): Higher is better
   │  └─ Peak indicates best clustering quality

4. ML_MODEL_COMPARISON.PNG
   └─ Which classifier scored highest?
   └─ All metrics should be similar (no overfitting)
   └─ Consistency across folds shows robustness

5. FUZZY_CMEANS_ANALYSIS.PNG
   ├─ LEFT: Cluster assignments plotted
   │  └─ How separated are your groups?
   ├─ RIGHT: Membership heatmap
   │  └─ Red = strong membership
   │  └─ White = weak membership

6. FUZZY_MEMBERSHIP_FUNCTIONS.PNG
   ├─ LEFT: Input membership (Low/Medium/High)
   ├─ MIDDLE: Output membership (Poor/Fair/Good/Excellent)
   └─ RIGHT: Inference result

7. SEM_FACTOR_LOADINGS.CSV
   └─ Open in Excel: Which variables matter most?
   └─ High values (>0.6) = strong contributors
"""

# ============================================================================
# STEP 5: UNDERSTAND KEY METRICS
# ============================================================================

"""
STRUCTURAL EQUATION MODELING:
  • Factor Loading: Strength of variable-factor relationship (0-1)
    - >0.6 = Strong, 0.4-0.6 = Moderate, <0.4 = Weak
  • Variance Explained: % of data variation captured by factors
    - >60% = Good, 40-60% = Moderate, <40% = Poor
  • Correlation: Linear relationship between variables (-1 to +1)
    - -1 = Perfect negative, 0 = No relationship, +1 = Perfect positive

MACHINE LEARNING CLASSIFICATION:
  • Accuracy: Percentage of correct predictions (0-1)
    - >0.85 = Excellent, 0.7-0.85 = Good, <0.7 = Poor
  • Precision: Of positive predictions, how many were correct?
    - High precision = few false positives
  • Recall: Of actual positives, how many were found?
    - High recall = few false negatives
  • F1-Score: Harmonic mean of precision & recall
    - Best overall metric for imbalanced data
  • Feature Importance: Which variables drive the model?
    - Sum to 100%, top 3-5 should be interpretable

FUZZY DATA ANALYSIS:
  • Fuzzy Partition Coefficient (FPC): Clustering quality (0-1)
    - >0.8 = Excellent clusters
    - 0.6-0.8 = Good clusters
    - <0.6 = Overlapping clusters
  • Membership Degree: Probability of belonging to cluster (0-1)
    - >0.7 = Strong membership
    - 0.3-0.7 = Uncertain/overlapping
    - <0.3 = Weak membership
"""

# ============================================================================
# STEP 6: ADVANCED CUSTOMIZATION
# ============================================================================

"""
MODIFY THE ANALYSIS FOR YOUR NEEDS:

A. CHANGE DATA SOURCE:
   Edit this line in analysis_sem_ml_fuzzy.py:
   
   excel_file = "YOUR_FILE_NAME.xlsx"

B. USE SPECIFIC SHEET:
   In load_and_preprocess_data() function, change:
   
   df = pd.read_excel(excel_path, sheet_name=0)
   # 0 = first sheet, 1 = second sheet, etc.
   # Or use sheet name: sheet_name='Data'

C. SELECT SPECIFIC COLUMNS:
   After loading, add:
   
   df = df[['Column1', 'Column2', 'Column3']]

D. ADJUST CLUSTER NUMBERS (for ML):
   In perform_ml_classification(), modify:
   
   K_range = range(2, 10)  # Explore 2-10 clusters

E. CHANGE FUZZY CLUSTER COUNT:
   In perform_fuzzy_analysis(), modify:
   
   n_clusters = 5  # Default is 3

F. HANDLE OUTLIERS:
   In load_and_preprocess_data(), add before return:
   
   from scipy import stats
   z_scores = np.abs(stats.zscore(df))
   df = df[(z_scores < 3).all(axis=1)]  # Remove extreme values

G. EXPORT CLUSTER ASSIGNMENTS:
   In perform_ml_classification(), add after clustering:
   
   pd.DataFrame({
       'Respondent_ID': df.index,
       'Cluster': cluster_labels
   }).to_csv('cluster_assignments.csv', index=False)
"""

# ============================================================================
# STEP 7: TROUBLESHOOTING
# ============================================================================

"""
PROBLEM: "ModuleNotFoundError: No module named 'pandas'"
SOLUTION: 
  pip install pandas
  (Or reinstall all: pip install -r requirements.txt)

PROBLEM: "FileNotFoundError: revised_coding_cc_data_05082025.xlsx"
SOLUTION:
  1. Check Excel file is in same folder as script
  2. Verify filename spelling matches exactly
  3. Edit excel_file variable in script if different name

PROBLEM: Script runs but produces no visualizations
SOLUTION:
  1. Check for error messages at end of output
  2. Ensure matplotlib is installed: pip install matplotlib
  3. Check write permissions in current directory

PROBLEM: "MemoryError" with large dataset
SOLUTION:
  Sample the data first in load_and_preprocess_data():
  
  df = df.sample(n=5000, random_state=42)

PROBLEM: ML clustering shows only 1 cluster
SOLUTION:
  Data might not have natural clusters. Try:
  1. Expand K_range to range(2, 20)
  2. Check data distribution (all values too similar?)
  3. Try different scaling methods

PROBLEM: Fuzzy analysis slow or produces warnings
SOLUTION:
  Reduce n_clusters or increase error threshold:
  
  fuzz.cluster.cmeans(..., error=0.01, maxiter=50)
"""

# ============================================================================
# STEP 8: EXAMPLE INTERPRETATION
# ============================================================================

"""
SCENARIO: Cultural Competence Scale Analysis

DATA: 150 respondents answered 20-item questionnaire (1-5 scale)

RESULTS INTERPRETATION:

✓ SEM Factor Analysis found 3 latent factors:
  - Factor 1: "Cultural Awareness" (questions 1-7 load >0.7)
  - Factor 2: "Cultural Knowledge" (questions 8-14 load >0.7)
  - Factor 3: "Cultural Skills" (questions 15-20 load >0.7)
  → Validates questionnaire structure! ✓

✓ ML Classification (Random Forest best):
  - Accuracy: 0.87 (classified correctly 87% of time)
  - Top 3 features: Q5, Q10, Q18
  - 4 distinct respondent groups identified
  → Can segment respondents for targeted intervention ✓

✓ Fuzzy Analysis (FPC=0.72):
  - Clear but slightly overlapping clusters
  - Some respondents belong to 2+ groups (10-20% membership)
  → Groups exist but with natural uncertainty ✓

RECOMMENDATIONS:
  1. Use 3-factor model for future surveys
  2. Focus competence training on Q5, Q10, Q18
  3. Design group-specific interventions for 4 segments
  4. Accept fuzzy boundaries when assigning new respondents
"""

# ============================================================================
# STEP 9: NEXT STEPS
# ============================================================================

"""
AFTER YOUR FIRST ANALYSIS:

1. VALIDATE FINDINGS:
   - Compare results with domain experts
   - Review outliers in cluster assignments
   - Check if clusters match expected patterns

2. REFINE THE MODEL:
   - Exclude questions with low loadings (<0.4)
   - Retrain with refined questionnaire
   - Compare new vs old metrics

3. USE FOR PREDICTIONS:
   - Export best ML model
   - Predict clusters for new respondents
   - Track changes over time

4. PUBLISH RESULTS:
   - Cite factor structure as evidence of validity
   - Report classification accuracy
   - Describe fuzzy memberships in uncertainty

5. AUTOMATE WORKFLOW:
   - Create scheduled analysis pipeline
   - Monitor changes in respondent profiles
   - Alert on significant pattern changes
"""

# ============================================================================
# STEP 10: QUICK REFERENCE COMMANDS
# ============================================================================

"""
# Run full analysis (recommended)
python analysis_sem_ml_fuzzy.py

# Run with specific data file
# Edit excel_file variable, then: python analysis_sem_ml_fuzzy.py

# Check Python version (must be 3.7+)
python --version

# List installed packages
pip list

# Upgrade a package
pip install --upgrade pandas

# Uninstall everything and reinstall
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# View installed package version
python -c "import pandas; print(pandas.__version__)"
"""

# ============================================================================
# ADDITIONAL RESOURCES
# ============================================================================

"""
LEARNING MATERIALS:

Structural Equation Modeling:
  • Hair, J. F. et al. (2010). Multivariate Data Analysis (7th ed.)
  • https://en.wikipedia.org/wiki/Structural_equation_modeling

Machine Learning:
  • James, G., et al. (2013). Introduction to Statistical Learning
  • https://scikit-learn.org/stable/supervised_learning.html

Fuzzy Logic:
  • Klir, G. J., & Yuan, B. (1995). Fuzzy Sets and Fuzzy Logic
  • https://en.wikipedia.org/wiki/Fuzzy_logic

Python Libraries:
  • Pandas: https://pandas.pydata.org/docs/
  • Scikit-learn: https://scikit-learn.org/stable/
  • Matplotlib: https://matplotlib.org/stable/api/
  • Seaborn: https://seaborn.pydata.org/api.html
"""

print(__doc__)
