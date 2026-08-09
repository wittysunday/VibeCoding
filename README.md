# VibeCoding Data Analysis Suite

Comprehensive Python implementation for **Structural Equation Modeling (SEM)**, **Machine Learning Classification**, and **Fuzzy Data Analysis** of questionnaire and survey data.

## Overview

This project provides an integrated analysis pipeline for processing Cultural Competence Scale questionnaire data with three complementary analytical approaches:

### 1. **Structural Equation Modeling (SEM)**
- Factor Analysis to identify underlying latent structures
- Correlation analysis of measurement constructs
- Factor loadings and variance explained
- Ideal for validating questionnaire reliability

### 2. **Machine Learning Classification**
- Automated clustering (K-Means with silhouette optimization)
- Multiple classifier comparison (Logistic Regression, Random Forest, Gradient Boosting)
- Feature importance analysis
- Cross-validation and performance metrics (Accuracy, Precision, Recall, F1-Score)

### 3. **Fuzzy Data Analysis**
- Fuzzy C-Means clustering for soft assignments
- Membership degree analysis
- Fuzzy logic inference system with linguistic variables
- Handles uncertainty in questionnaire responses

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup

1. **Clone or download this repository**
   ```bash
   git clone https://github.com/wittysunday/VibeCoding.git
   cd VibeCoding
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Execution

```bash
python analysis_sem_ml_fuzzy.py
```

The script will automatically:
- Load `revised_coding_cc_data_05082025.xlsx`
- Perform preprocessing and normalization
- Execute all three analytical pipelines
- Generate visualizations and reports

### Input Data Requirements

Place your Excel file in the same directory as the script. Expected format:
- **File**: `revised_coding_cc_data_05082025.xlsx`
- **Content**: Questionnaire responses with numeric values
- **Minimum**: 3+ numeric columns, 30+ rows recommended

### Output Files

#### Visualizations (PNG)
| File | Description |
|------|-------------|
| `sem_correlations.png` | Heatmap of variable correlations |
| `ml_clustering_analysis.png` | Elbow method and silhouette score plots |
| `ml_model_comparison.png` | Performance metrics for all classifiers |
| `fuzzy_cmeans_analysis.png` | Fuzzy clusters and membership heatmap |
| `fuzzy_membership_functions.png` | Membership functions for fuzzy logic |

#### Data Files (CSV)
| File | Description |
|------|-------------|
| `sem_factor_loadings.csv` | Factor loadings and variable contributions |

#### Reports (TXT)
| File | Description |
|------|-------------|
| `analysis_report.txt` | Summary of all analyses and findings |

## Detailed Analysis Components

### Structural Equation Modeling (SEM)

**Purpose**: Validate questionnaire structure and identify latent constructs

**Features**:
- Automatic standardization of variables
- Factor Analysis with optimal component selection
- Correlation matrix computation
- Variance decomposition

**Key Outputs**:
- Factor loadings (which variables load onto each factor)
- Explained variance ratios
- Correlation heatmap visualization

**Interpretation**:
- High loadings (>0.6) indicate strong relationships
- Low explained variance suggests multiple independent constructs
- Correlations between factors indicate construct validity

### Machine Learning Classification

**Purpose**: Discover patterns and classify respondents into meaningful groups

**Features**:
- Optimal cluster detection using silhouette score
- Three classifier algorithms:
  - **Logistic Regression**: Linear boundaries, interpretable
  - **Random Forest**: Non-linear patterns, feature importance
  - **Gradient Boosting**: Highest accuracy, ensemble approach
- 5-fold cross-validation
- Feature importance ranking

**Key Outputs**:
- Cluster assignments for each respondent
- Performance metrics for each model
- Top features driving classification
- Model comparison visualization

**Use Cases**:
- Segment respondents by competence level
- Predict competence category for new respondents
- Identify key differentiating factors

### Fuzzy Data Analysis

**Purpose**: Handle uncertainty and vagueness in questionnaire responses

**Features**:
- Fuzzy C-Means clustering with soft membership
- Linguistic variable definitions (Low, Medium, High)
- Fuzzy inference system simulation
- Membership degree analysis

**Key Outputs**:
- Fuzzy Partition Coefficient (FPC) - quality metric (0-1, higher is better)
- Membership matrix showing degree of belonging to each cluster
- Fuzzy logic membership function plots

**Use Cases**:
- Handle overlapping response categories
- Model linguistic uncertainty ("somewhat agree")
- Soft clustering where samples can belong to multiple groups

## Customization Guide

### Modify Data Loading

Edit the `load_and_preprocess_data()` function to:
- Change which sheet to load: `sheet_name=1` for second sheet
- Adjust missing value thresholds
- Apply domain-specific preprocessing

```python
df = pd.read_excel(excel_path, sheet_name='Sheet_Name')
```

### Adjust SEM Factor Structure

Modify the descriptor in `perform_sem_analysis()`:

```python
# Example: Define specific latent variables
desc = """
competence ~ cultural_awareness + cultural_knowledge + cultural_skills
satisfaction ~ competence
"""
```

### Tune ML Parameters

In `perform_ml_classification()`:

```python
# Change optimal cluster range
K_range = range(2, 10)  # Explore more clusters

# Adjust classifier parameters
RandomForestClassifier(n_estimators=200, max_depth=15, ...)
```

### Configure Fuzzy Clustering

In `perform_fuzzy_analysis()`:

```python
n_clusters = 5  # Change number of fuzzy clusters
m = 2.5  # Fuzziness parameter (higher = softer clusters)
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation and I/O |
| `numpy` | Numerical operations |
| `scikit-learn` | ML algorithms and preprocessing |
| `matplotlib` | Visualization backend |
| `seaborn` | Statistical visualizations |
| `scikit-fuzzy` | Fuzzy logic and C-Means |
| `semopy` | Structural equation modeling |
| `scipy` | Scientific computing utilities |
| `openpyxl` | Excel file handling |

## Example Workflow

```python
from analysis_sem_ml_fuzzy import *

# 1. Load data
df = load_and_preprocess_data('revised_coding_cc_data_05082025.xlsx')

# 2. Run SEM analysis
sem_results = perform_sem_analysis(df)

# 3. Run ML classification
ml_results = perform_ml_classification(df)

# 4. Run fuzzy analysis
fuzzy_results = perform_fuzzy_analysis(df)

# 5. Generate comprehensive report
generate_analysis_report(sem_results, ml_results, fuzzy_results)
```

## Troubleshooting

### "File not found" Error
```
Solution: Ensure Excel file is in current directory and correctly named
```

### ImportError for scikit-fuzzy
```bash
pip install scikit-fuzzy --upgrade
```

### Memory Error with Large Datasets
```python
# Reduce data sample size
df = df.sample(n=5000, random_state=42)
```

### Clustering Results Unclear
```python
# Adjust K_range in perform_ml_classification()
K_range = range(2, 15)  # Wider exploration
```

## Interpretation Guide

### Good SEM Results
- Factor loadings: 0.6 - 0.9
- Explained variance: >60%
- Correlations between constructs: -0.3 to 0.7

### Good ML Results
- F1-Score: >0.7
- Cross-validation consistency: σ < 0.1
- Feature importance concentrated in 3-5 features

### Good Fuzzy Results
- FPC: >0.6
- Membership entropy: 0.5-0.8
- Clear cluster separation with soft boundaries

## Research Applications

This analysis suite is ideal for:
- **Questionnaire Validation**: Confirm measurement reliability (SEM)
- **Respondent Segmentation**: Identify distinct groups (ML + Fuzzy)
- **Pattern Discovery**: Understand relationships between variables (SEM)
- **Predictive Analytics**: Classify new respondents (ML)
- **Uncertainty Modeling**: Handle vague or overlapping responses (Fuzzy)

## References

- Hair et al. (2010). Multivariate Data Analysis
- James et al. (2013). An Introduction to Statistical Learning
- Klir & Yuan (1995). Fuzzy Sets and Fuzzy Logic

## License

MIT License - Feel free to use and modify for research and commercial applications.

## Author

Created for the VibeCoding project - Cultural Competence Data Analysis

---

**Questions or Issues?** Open an issue on GitHub or contact the project maintainer.
