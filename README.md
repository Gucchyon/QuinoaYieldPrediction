# Quinoa Yield Prediction Using UAV-Derived Features

This repository contains the complete machine learning pipeline for predicting quinoa yield using UAV-derived features across different growth stages, as described in Sesay et al. (2026).

## Overview

This notebook implements a comprehensive analysis of quinoa yield prediction using:
- Multiple machine learning algorithms (Ridge Regression, K-Nearest Neighbors, Random Forest, XGBoost, LightGBM)
- Nested cross-validation for robust model evaluation
- Feature importance analysis using SHAP and permutation importance
- Multi-growth stage analysis to identify optimal prediction timing

## Dataset

The analysis uses data from 380 individual quinoa plants across:
- **Seasons**: 2022 (80 plants) and 2024-2025 (300 plants)
- **Varieties**: 8 quinoa varieties (CA04, J027, J040, J075, J082, J100, Kd, NL6)
- **Growth Stages**: Early Vegetative, Late Vegetative, Flowering, Grain Filling, Maturity
- **Features**: 16 UAV-derived features including:
  - 3D structural parameters: Plant Height (PH), Plant Surface Area (PSA), Plant Volume (PV)
  - Spectral vegetation indices: DVI, RVI, RDVI, MSAVI, MSAVI2, OSAVI, EVI, NDVI, SAVI, SARE, NIRREDVI, RENDVI, RERVI

## Requirements

The analysis was conducted in the following computational environment:

### Python Version
- Python 3.12.12 (main, Oct 10 2025, 08:52:57) [GCC 11.4.0]

### Package Versions
- scikit-learn: 1.6.1
- XGBoost: 3.1.2
- LightGBM: 4.6.0
- NumPy: 2.0.2
- pandas: 2.2.2
- Matplotlib: 3.10.0
- Seaborn: 0.13.2
- SHAP: 0.50.0

### Installation

```bash
pip install scikit-learn==1.6.1 xgboost==3.1.2 lightgbm==4.6.0 numpy==2.0.2 pandas==2.2.2 matplotlib==3.10.0 seaborn==0.13.2 shap==0.50.0
```

## Notebook Structure

The notebook is organized into the following sections:

### 1. Setup
- Environment information
- Data loading and preprocessing
- Feature and target variable definition

### 2. Correlation Analysis
- Pearson correlation analysis between features
- Visualization of correlation matrices across growth stages

### 3. Modeling

#### Nested Cross-Validation
- **Outer Loop**: 5-fold cross-validation for model evaluation (80/20 train-test split)
- **Inner Loop**: 3-fold cross-validation for hyperparameter tuning
- **Random States**: Fixed for reproducibility (outer: 42, inner: 43)
- **Model Selection**: Best model selected based on lowest average test RMSE

#### Hyperparameter Search Spaces
- **Ridge Regression**: 20 logarithmically-spaced alpha values (10⁻⁴ to 10⁰)
- **K-Nearest Neighbors**: n_neighbors (1 to 20)
- **Random Forest**: n_estimators (50, 100, 200), max_depth (None, 10, 20), min_samples_split (2, 5, 10)
- **XGBoost**: n_estimators (100, 200, 300), learning_rate (0.01, 0.1, 0.2), max_depth (3, 5, 7)
- **LightGBM**: n_estimators (100, 200, 300), learning_rate (0.01, 0.1, 0.2), num_leaves (30, 50, 100)

#### Evaluation Metrics
- **R²**: Coefficient of determination (prediction accuracy)
- **RMSE**: Root mean square error (error magnitude)
- **MAE**: Mean absolute error (average prediction deviation)

### 4. Feature Importance Analysis

#### SHAP (SHapley Additive exPlanations)
- Quantifies marginal contribution of each feature to model predictions
- Uses TreeExplainer for tree-based models and KernelExplainer for others
- Mean absolute SHAP values calculated across all test folds

#### Permutation Importance
- Measures model performance degradation when feature-target relationships are disrupted
- 10 random permutations per feature (random_state=42)
- Scoring metric: R²
- Final values averaged across 5 cross-validation folds

### 5. Visualization
- Model performance comparison across growth stages
- Scatter plots of predicted vs. actual yield
- Feature importance heatmaps (SHAP and Permutation Importance)
- Correlation matrices

## Usage

### Running in Google Colaboratory

1. Upload your quinoa dataset CSV file when prompted
2. The CSV file should contain the following columns:
   - `Growth Stage`: Growth stage label
   - `Variety`: Quinoa variety identifier
   - `Replication`: Experimental replication number
   - `Sample`: Sample identifier
   - Feature columns: PH, PSA, PV, DVI, RVI, RDVI, MSAVI, MSAVI2, OSAVI, EVI, NDVI, SAVI, SARE, NIRREDVI, RENDVI, RERVI
   - Target column: `Yld g/plant` (yield in grams per plant)

3. Execute cells sequentially to:
   - Load and preprocess data
   - Perform correlation analysis
   - Train and evaluate models using nested cross-validation
   - Calculate feature importance using SHAP and permutation importance
   - Generate visualizations

### Expected Outputs

The notebook generates the following outputs in the `/content/figures` directory:
- `correlation_heatmap.png`: Correlation matrix visualization
- `model_performance_comparison.png`: Bar charts comparing model performance
- `scatter_plots.png`: Predicted vs. actual yield plots for best models
- `combined_feature_importance.png`: SHAP and permutation importance heatmaps

Trained models are saved in `/content/models` as pickle files for each growth stage.

## Key Findings

The analysis reveals:
- **Optimal Prediction Timing**: Grain filling stage provides the best prediction accuracy
- **Feature Importance Patterns**: 
  - Early Vegetative: Spectral indices (MSAVI, NDVI) dominate
  - Late Vegetative onwards: 3D structural features (PSA, PV, PH) become increasingly important
- **Model Performance**: Tree-based ensemble methods (Random Forest, XGBoost, LightGBM) generally outperform linear models
- **Convergence of Methods**: SHAP and permutation importance analyses show consistent patterns, validating the robustness of identified features

## Citation

If you use this code, please cite:

```
Sesay et al. (2026). [Title of the paper]. [Journal name].
```

*Note: Full citation details will be updated upon publication.*

## License

This project is licensed under the MIT License - see below for details.

```
MIT License

Copyright (c) 2025 Guchyos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Contact

For questions or issues, please contact [your contact information].

## Acknowledgments

This research was conducted using UAV-based phenotyping data collected across two growing seasons. We acknowledge the contributions of all team members involved in data collection and analysis.

