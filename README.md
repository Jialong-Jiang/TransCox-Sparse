
# TransCox-Sparse: 高维稀疏生存分析的迁移学习R包

[![R](https://img.shields.io/badge/R-%3E%3D3.6.0-blue.svg)](https://www.r-project.org/)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.0-orange.svg)](https://tensorflow.org/)

## 📖 简介

TransCox-Sparse是TransCox包的增强版本，专门为高维稀疏生存分析设计。它通过迁移学习技术，利用源域（辅助）数据来改善目标域（主要）数据的Cox回归模型性能，特别适用于高维数据（p >> n）的特征选择和稀疏建模。
项目初衷是由于发现TransCox本身并不能很好的处理高维稀疏情况问题，由于本人技术不精，绝大多数代码均为AI辅助编写。

### 🌟 主要特性

- **🔄 迁移学习**: 从源域数据中学习知识来改善目标域模型
- **📊 高维数据支持**: 处理特征数远大于样本数的情况（p >> n）
- **✨ 稀疏性建模**: 通过L1正则化和软阈值化实现特征选择
- **🤖 自动调参**: 基于BIC的超参数自动选择
- **🔙 向后兼容**: 完全兼容原始TransCox功能
- **⚡ 高效优化**: 基于TensorFlow的梯度优化算法

## 🚀 快速开始

### 环境要求

- R >= 3.6.0
- Python >= 3.7
- TensorFlow 2.10.0

### 安装步骤 优先参考官方文档：[TransCox GitHub Repository](https://github.com/ziyili20/TransCox)

1. **Python环境配置**
```bash
# 创建conda环境
conda create -n TransCoxEnvi python=3.8
conda activate TransCoxEnvi

# 安装必要包
conda install tensorflow=2.10.0
conda install numpy pandas
```

2. **R环境配置**
```r
# 安装必要的R包
install.packages(c("survival", "glmnet", "Matrix", "reticulate"))

# 配置Python环境
library(reticulate)
use_condaenv("TransCoxEnvi")
```

3. **下载和使用**
```r
# 克隆仓库
git clone https://github.com/Jialong-Jiang/TransCox-Sparse.git
cd TransCox-Sparse

# 设置工作目录并加载函数
setwd("path/to/TransCox-Sparse")
source("R/runTransCox_Sparse.R")
source_python("inst/python/TransCoxFunction_Sparse.py")
```

## 📋 使用示例

### 基础使用

```r
# 加载必要库
library(survival)
library(Matrix)
library(reticulate)

# 高维稀疏数据分析
result <- runTransCox_Sparse(
    primData = your_target_data,
    auxData = your_source_data,
    cov = paste0("X", 1:500),  # 500个特征
    statusvar = "status",
    lambda_beta = 0.05,        # 稀疏性控制
    auto_tune = TRUE,          # 自动调参
    verbose = TRUE
)

# 查看稀疏性结果
nonzero_coef <- sum(abs(result$new_beta) > 1e-8)
sparsity <- (1 - nonzero_coef / length(result$new_beta)) * 100
cat("稀疏度:", round(sparsity, 1), "%\n")
```

### 自动参数调优

```r
# 完全自动化分析
result_auto <- runTransCox_Sparse(
    primData = primData,
    auxData = auxData,
    cov = feature_names,
    lambda1 = NULL,      # 自动选择
    lambda2 = NULL,      # 自动选择
    lambda_beta = NULL,  # 自动选择
    auto_tune = TRUE
)
```

## 📊 超参数指南

### 稀疏性控制参数 `lambda_beta`

| 稀疏度目标 | 推荐值 | 适用场景 |
|------------|--------|----------|
| 0% (无稀疏) | 0 | 标准TransCox |
| 10-30% | 0.01-0.03 | 轻度特征选择 |
| 30-60% | 0.05-0.08 | 中度特征选择 |
| 60-90% | 0.1-0.15 | 强特征选择 |
| >90% | 0.2+ | 极度稀疏 |

### 数据类型推荐参数

| 数据类型 | 特征数 | lambda1 | lambda2 | lambda_beta | learning_rate |
|----------|--------|---------|---------|-------------|---------------|
| 低维 | p < 50 | 0.1 | 0.1 | 0.01 | 0.004 |
| 中维 | 50 ≤ p ≤ 500 | 0.05 | 0.05 | 0.05 | 0.002 |
| 高维 | p > 500 | 0.01 | 0.01 | 0.1 | 0.001 |

## 📁 项目结构

```
TransCox-Sparse/
├── R/                          # R函数
│   ├── runTransCox_Sparse.R   # 主要接口函数
│   ├── SelParam_By_BIC_Sparse.R # 稀疏参数选择
│   └── GetAuxSurv_Sparse.R    # 稀疏源域分析
├── inst/python/               # Python优化引擎
│   ├── TransCoxFunction.py    # 原始优化函数
│   └── TransCoxFunction_Sparse.py # 稀疏优化函数
├── tests/                     # 测试文件
│   ├── test_sparse_effect.R   # 稀疏效果测试
│   └── test_integration.R     # 集成测试
├── TransCox_用户手册.md       # 详细用户手册
└── README.md                  # 本文件
```

## 🔬 算法原理

### 1. 原始 TransCox 模型

#### 优化目标
原始 TransCox 模型通过迁移学习从源域（auxiliary cohort）调整到目标域（primary cohort），优化目标为：
$$\min L(\delta, \xi) + \lambda_1 |\delta|_1 + \lambda_2 |\xi|_1$$

#### 具体参数及含义
- $L(\delta, \xi)$: 目标域的联合负对数似然，捕捉系数差异和基线风险调整的拟合效果：
  $$L(\delta, \xi) = -\sum_{i \in target} [X_i \beta_s + X_i \delta - \log \sum_{j \in R(t_i)} \exp(X_j \beta_s + X_j \delta)] + \sum \xi(t_i)$$
- $X_i$: 第 $i$ 个样本的协变量向量。
- $\beta_s$: 源域系数（由 `coxph` 估计）。
- $\delta$: 系数差异，调整 $\beta_s$ 到目标域系数 $\beta_t = \beta_s + \delta$。
- $R(t_i)$: 时间 $t_i$ 的风险集。
- $\xi(t_i)$: 基线风险调整，捕捉源域基线风险 $h_{0s}(t)$ 到目标域 $h_{0t}(t) = h_{0s}(t) + \xi(t)$ 的时间变化异质性。
- $\lambda_1 |\delta|_1$: L1 惩罚，诱导 $\delta$ 稀疏，确保只调整关键协变量的效应。
- $\lambda_2 |\xi|_1$: L1 惩罚，诱导 $\xi(t)$ 稀疏，确保只调整关键时间点的基线风险。
- $\lambda_1, \lambda_2$: 正则化参数，通过 BIC 调参，选择最小 BIC 值以平衡拟合度和稀疏性。

#### 含义
模型通过优化 $\delta$ 和 $\xi(t)$，从源域借用信息，调整到目标域，处理时间变化异质性（如不同医疗队列的治疗效果差异）。
L1 惩罚确保稀疏迁移，只调整重要协变量和时间点，适合低维场景（$p \ll n$，如论文中的 $p=5$）。
优化目标通过联合似然捕捉目标域的生存分布，同时利用源域信息提高估计精度。

### 2. 高维稀疏 TransCox_sparse 模型

#### 优化目标
为处理高维稀疏数据（如基因组数据，$p \gg n$），TransCox_sparse 模型扩展优化目标为：
$$\min L(\delta, \xi) + \lambda_1 |\delta|_1 + \lambda_2 |\xi|_1 + \lambda_\beta |\beta_s + \delta|_1$$

#### 具体参数及含义
- $L(\delta, \xi)$: 同原始模型，目标域负对数似然，捕捉 $\delta$ 和 $\xi(t)$ 的拟合效果。
- $\lambda_1 |\delta|_1$: 控制 $\delta$ 的稀疏性，确保迁移差异集中在少量关键协变量。
- $\lambda_2 |\xi|_1$: 控制 $\xi(t)$ 的稀疏性，确保基线风险调整集中在关键时间点。
- $\lambda_\beta |\beta_s + \delta|_1$: 新增 L1 惩罚，作用于目标域系数 $\beta_t = \beta_s + \delta$，诱导 $\beta_t$ 整体稀疏，适合高维场景（例如，$p=1000$ 时，非零系数目标 < 50）。
- $\lambda_\beta$: 正则化参数，与 $\lambda_1, \lambda_2$ 一起通过 BIC 调参。
- $\beta_s$: 源域系数，改进为通过 Lasso-Cox 估计：
  $$\min - l(\beta_s) + \lambda_s |\beta_s|_1$$
  确保 $\beta_s$ 本身稀疏，减少高维下的过拟合。

#### 含义
- **新增 $\lambda_\beta |\beta_s + \delta|_1$**: 原始模型仅惩罚 $\delta$，无法保证 $\beta_t$ 整体稀疏（尤其当 $\beta_s$ 非稀疏时）。新增项直接约束 $\beta_t$，类似 Lasso-Cox 的稀疏性，适合高维稀疏数据（如基因数据，只有少数特征相关）。
- **稀疏性效果**: 通过软阈值 $S(z, \lambda) = \text{sign}(z) \max(|z| - \lambda, 0)$，收缩 $\beta_t$ 的小值分量为 0，减少非零系数，降低过拟合风险。
- **高维支持**: 支持稀疏矩阵输入（如 `dgCMatrix`），减少内存占用，适合基因组等高维场景。

### 3. 实现优化的方法

#### 源域改进
- **原始方法**: `GetAuxSurv.R` 使用 `coxph` 估计 $\beta_s$，高维 ($p \gg n$) 时不稳定或过拟合。
- **改进方法**: 使用 `glmnet` 实现 Lasso-Cox 估计 $\beta_s$：
  - 确保 $\beta_s$ 稀疏（非零系数接近真实信号，如 20）。
  - 通过交叉验证选择 $\lambda_s$，提高高维稳定性。

#### BIC 调参
- **原始方法**: BIC 调 $\lambda_1, \lambda_2$，自由度 $df = \text{非零}(\delta) + \text{非零}(\xi)$：
  $$\text{BIC} = -2 \log L + \log(n) \cdot df$$
  网格搜索 $\lambda_1, \lambda_2$，选择最小 BIC。
- **改进方法**: 扩展 BIC 包含 $\lambda_\beta$：
  - 自由度：$df = \text{非零}(\delta) + \text{非零}(\xi) + \text{非零}(\beta_s + \delta)$。
  - 网格 search $\lambda_1, \lambda_2, \lambda_\beta$，生成 3D BIC 阵列：
  - 选择最小 BIC 的参数组合，确保稀疏性和拟合度平衡。

## 📚 文档

- [原始TransCox论文](https://www.tandfonline.com/doi/full/10.1080/01621459.2023.2210336) -


## 🧪 demo测试

```r
# 运行稀疏效果测试
source("demo_transcox_optimal.R")

```

## 🤝 贡献

欢迎提交Issue和Pull Request！



## 📄 许可证

本项目采用 [MIT License](LICENSE)

## 📞 联系

如有问题或建议，请：
- 提交 [Issue](https://github.com/Jialong-Jiang/TransCox-Sparse/issues)
- 发送邮件至: 2672159435@qq.com

## 🙏 致谢

- 原始TransCox包的开发者
- TensorFlow和R社区
- 所有贡献者和用户

---

**⭐ 如果这个项目对您有帮助，请给我们一个Star！**
