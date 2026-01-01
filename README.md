# TransCox-Sparse: Transfer Learning R Package for High-Dimensional Sparse Survival Analysis

[![R](https://img.shields.io/badge/R-%3E%3D4.4.3-blue.svg)](https://www.r-project.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-orange.svg)](https://tensorflow.org/)

## 📖 Introduction

TransCox-Sparse is an enhanced version of the TransCox package, designed for high-dimensional sparse survival analysis. It uses transfer learning to improve Cox regression model performance on a target (primary) domain by leveraging source (auxiliary) domain data, especially suitable for feature selection and sparse modeling in high-dimensional settings (p >> n).

The project was initiated because the original TransCox did not handle high-dimensional sparse situations well.

### 🌟 Key Features

- **🔄 Transfer Learning**: Learn from source domain data to improve target domain models
- **📊 High-Dimensional Support**: Handle cases where features far exceed samples (p >> n)
- **✨ Sparse Modeling**: Feature selection via L1 regularization and soft thresholding
- **🤖 Auto-Tuning**: BIC-based automatic hyperparameter selection
- **🔙 Backward Compatibility**: Fully compatible with original TransCox functions

## 🚀 Quick Start

### Requirements

- R >= 4.4.3
- Python >= 3.10
- TensorFlow 2.18.0

### Installation Steps

1. **Python Environment Setup**
```bash
# Create conda environment
conda create -n TransCoxEnvi python=3.10
conda activate TransCoxEnvi

# Install necessary packages
conda install tensorflow=2.18.0
conda install numpy pandas
```

2. **R Environment Setup**
```r
# Install necessary R packages
install.packages(c("survival", "glmnet", "Matrix", "reticulate"))

# Configure Python environment
library(reticulate)
use_condaenv("TransCoxEnvi")
use_python("D:/anaconda3/envs/TransCoxEnvi/python.exe", required = TRUE)#change to your dir
 

```

3. **Download and Usage**
```r
# Clone repository
if (!require("devtools")) install.packages("devtools")
devtools::install_github("Jialong-Jiang/TransCox-Sparse")
# if devtools fails try
if (!require("remotes")) install.packages("remotes")
remotes::install_github("Jialong-Jiang/TransCox-Sparse")


# Set working directory and load functions
library(reticulate)
source_python(system.file("python", "TransCoxFunction_Sparse.py", package = "TransCoxSparse"))
```

4. **Demos**

```r
library(TransCoxSparse)
source_python(system.file("python", "TransCoxFunction_Sparse.py", package = "TransCoxSparse"))
# Run a single demonstration with visualizations
demo("quick_start", package = "TransCoxSparse")  # Simple demo takes about 3-5 minutes
demo("reproduce_simulation.R", package = "TransCoxSparse")  # 100-run Monte Carlo simulation takes about 1.5h 

# You can skip to the precomuted results by running the following code

# View pre-computed simulation results from the paper
demo("view_precomputed_results", package = "TransCoxSparse")
```

## 📋 Usage Examples

### Basic Usage

```r
# Load necessary libraries
library(survival)
library(Matrix)
library(reticulate)

# High-dimensional sparse data analysis
result <- runTransCox_Sparse(
    primData = your_target_data,
    auxData = your_source_data,
    cov = paste0("X", 1:500),  # 500 features
    statusvar = "status",
    lambda_beta = 0.05,        # Sparsity control
    auto_tune = TRUE,          # Auto tuning
    verbose = TRUE
)

# Check sparsity results
nonzero_coef <- sum(abs(result$new_beta) > 1e-8)
sparsity <- (1 - nonzero_coef / length(result$new_beta)) * 100
cat("Sparsity:", round(sparsity, 1), "%\n")
```

### Automatic Parameter Tuning

```r
# Fully automated analysis
result_auto <- runTransCox_Sparse(
    primData = primData,
    auxData = auxData,
    cov = feature_names,
    lambda1 = NULL,      # Auto selection
    lambda2 = NULL,      # Auto selection
    lambda_beta = NULL,  # Auto selection
    auto_tune = TRUE
)
```

## 📊 Hyperparameter Guide

### Sparsity Control Parameter `lambda_beta`

| Sparsity Target | Recommended Value | Applicable Scenario |
|-----------------|-------------------|---------------------|
| 0% (No sparsity) | 0 | Standard TransCox |
| 10-30% | 0.01-0.03 | Mild feature selection |
| 30-60% | 0.05-0.08 | Moderate feature selection |
| 60-90% | 0.1-0.15 | Strong feature selection |
| >90% | 0.2+ | Extreme sparsity |

### Recommended Parameters by Data Type

| Data Type | Features | lambda1 | lambda2 | lambda_beta | learning_rate |
|-----------|----------|---------|---------|-------------|---------------|
| Low-dim | p < 50 | 0.1 | 0.1 | 0.01 | 0.004 |
| Mid-dim | 50 ≤ p ≤ 500 | 0.05 | 0.05 | 0.05 | 0.002 |
| High-dim | p > 500 | 0.01 | 0.01 | 0.1 | 0.001 |

## 📁 Project Structure

```
TransCoxSparse/
├── demo/                                         # 🚀 Demo Scripts & Examples
│   ├── 00Index                                   # 🔖 Demo Index File
│   ├── quick_start.R                             # 🚀 Quick Start Demo (Single Run)
│   ├── reproduce_simulation.R                    # 🔁 Reproduce 100-Run Simulation
│   └── view_precomputed_results.R                # 👁️ View Pre-computed Results
├── inst/                                         # 📁 Installed Files (Python/Data)
│   ├── demo_results/                             # 📊 Precomputed Simulation Results 
│   │   ├── quick_start/                          # 📂
│   │   └── simulation_100/                       # 📂
│   └── python/                                   # 🐍 Python Optimization Engine
│       ├── TransCoxFunction.py                   # ⚡ Original Optimization Function 
│       └── TransCoxFunction_Sparse.py            # ⚡ Sparse Optimization Function 
├── man/                                          # 📘 R Documentation
│   ├── GetAuxSurv.Rd                             # 📘
│   └── ...                                       # 📘
├── R/                                            # 📁 R Function Library
│   ├── GetAuxSurv.R                              # 🔄 Source Domain Parameter Estimation 
│   ├── GetAuxSurv_Sparse.R                       # 🔄 Sparse Source Domain Analysis
│   ├── GetBIC.R                                  # 📈 BIC Calculation Function
│   ├── GetPrimaryParam.R                         # 🎯 Target Domain Parameter Estimation
│   ├── runTransCox_Sparse.R                      # 🎯 Selection Function 
│   ├── runTransCox_TwoStage.R                    # 🎯 Main Interface Function 
│   ├── SelParam_By_BIC_Sparse.R                  # 📊 Sparse Parameter Selection 
│   └── zzz_imports.R                             # 📦 Package Imports & Global Variables
│   └── ...
├── DESCRIPTION                                   # 📄 R Package Description File
├── NAMESPACE                                     # 📛 Export/Import Namespace
├── README.md                                     # 📖 Project Description
└── TransCoxSparse.Rproj                          # 📄
```

## 🔬 Algorithm Principles

### 1. Original TransCox Model

#### Optimization Objective
The original TransCox model adjusts from the source domain (auxiliary cohort) to the target domain (primary cohort) via transfer learning, with the optimization objective:

$$
\min L(\eta, \xi, \lambda_\eta, \lambda_\xi) = L(\eta, \xi) + \lambda_\eta \|\eta\|_1 + \lambda_\xi \|\xi\|_1
$$
Where
$$
L(\eta, \xi) = -\sum_{i=1}^N \left[ \delta_i \big(x_i^T (\hat{\beta}^s + \eta) + \log \Delta \hat{H}_0^s(y_i) + \xi_i\big) \right]+\sum_{i=1}^N \left[ \sum_{j=1}^{n_0} \big(\Delta \hat{H}_0^s(\tilde{y}_j)+\xi_j I(\tilde{y}_j \le y_i)\big)\exp\{x_i^T (\hat{\beta}^s + \eta)\} \right]
$$

#### Specific Parameters
- $L(\eta, \xi)$: Joint negative log-likelihood of the target domain, capturing coefficient differences and baseline hazard adjustment fit.
- $x_i$: Covariate vector for the i-th sample.
- $\hat{\beta}^s$: Source domain coefficients (estimated by `coxph`).
- $\eta$: Coefficient difference, adjusting $\hat{\beta}^s$ to target domain coefficients $\beta_t = \hat{\beta}^s + \eta$.
- $\Delta \hat{H}_0^s(y_i)$: Incremental baseline cumulative hazard at time $y_i$.
- $\xi_i$: Baseline hazard adjustment, capturing time-varying heterogeneity from source domain $h_{0s}(t)$ to target domain $h_{0t}(t) = h_{0s}(t) + \xi(t)$.
- $\lambda_\eta \|\eta\|_1$: L1 penalty, inducing sparsity in $\eta$, ensuring only key covariate effects are adjusted.
- $\lambda_\xi \|\xi\|_1$: L1 penalty, inducing sparsity in $\xi(t)$, ensuring baseline hazard adjustment only at key time points.
- $\lambda_\eta, \lambda_\xi$: Regularization parameters, selected via BIC to balance fit and sparsity.

#### Meaning
By optimizing $\eta$ and $\xi(t)$, the model borrows information from the source domain to adjust to the target domain, handling time-varying heterogeneity (e.g., treatment effect differences across cohorts).
L1 penalty ensures sparse transfer, adjusting only important covariates and time points. The objective captures the target domain's survival distribution via joint likelihood while leveraging source domain info for better estimation.

---

### 2. High-Dimensional Sparse TransCox_sparse Model

#### Optimization Objective
To handle high-dimensional sparse data (e.g., genomic data, $p >> n$), the TransCox_sparse model extends the optimization objective:

$$
\min L(\eta, \xi, \lambda_\eta, \lambda_\xi, \lambda_\beta)
= L(\eta, \xi) + \lambda_\eta \|\eta\|_1 + \lambda_\xi \|\xi\|_1 + \lambda_\beta \|\hat{\beta}^s + \eta\|_1
$$

#### Specific Parameters
- $L(\eta, \xi)$: Same as original model, target domain negative log-likelihood.
- $\lambda_\eta \|\eta\|_1$: Controls sparsity of $\eta$.
- $\lambda_\xi \|\xi\|_1$: Controls sparsity of $\xi(t)$.
- $\lambda_\beta \|\hat{\beta}^s + \eta\|_1$: New L1 penalty on target domain coefficients $\beta_t = \hat{\beta}^s + \eta$, inducing overall sparsity suitable for high-dimensional scenarios.
- $\lambda_\beta$: Regularization parameter, selected via BIC along with $\lambda_\eta, \lambda_\xi$.
- $\hat{\beta}^s$: Source domain coefficients, improved to be estimated via Lasso-Cox: $\min -l(\beta_s) + \lambda_s \Vert\beta_s\Vert_1$, ensuring $\beta_s$ itself is sparse to reduce overfitting in high dimensions.

#### Meaning
- **New term $\lambda_\beta \|\hat{\beta}^s + \eta\|_1$**: The original model only penalized $\eta$, unable to guarantee overall sparsity of $\beta_t$ (especially if $\hat{\beta}^s$ is not sparse). The new term directly constrains $\beta_t$, similar to Lasso-Cox sparsity.
- **Sparsity effect**: Shrinks small components of $\beta_t$ to 0 via soft thresholding $S(z, \lambda) = \text{sign}(z) \max(|z| - \lambda, 0)$, reducing overfitting.
- **High-dimensional support**: Supports sparse matrix input (e.g., `dgCMatrix`), reducing memory usage, suitable for genomic data.

## 📚 Documentation

- [Original TransCox Paper](https://www.tandfonline.com/doi/full/10.1080/01621459.2023.2210336) -

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📄 License

This project is licensed under the [MIT License](LICENSE)

## 📞 Contact

For questions or suggestions, please:
- Submit an [Issue](https://github.com/Jialong-Jiang/TransCox-Sparse/issues)
- Email to: 2672159435@qq.com

## 🙏 Acknowledgments

- Developers of the original TransCox package
- TensorFlow and R communities
- All contributors and users

---

**⭐ If this project helps you, please give us a Star!**
