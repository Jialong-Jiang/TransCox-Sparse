
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

### 稀疏性实现

1. **L1正则化**: 对最终系数β_t施加L1惩罚
2. **软阈值化**: 在每次梯度更新后应用软阈值操作
3. **自适应阈值**: 根据λ_beta动态调整阈值强度

### 优化目标

```
L(β) = -log-likelihood + λ₁||η||₁ + λ₂||ξ||₁ + λ_β||β_t||₁
```

其中：
- η: 系数变化项
- ξ: 风险变化项  
- β_t: 最终回归系数

## 📚 文档

- [详细用户手册](TransCox_用户手册.md) - 完整的使用指南和最佳实践
- [原始TransCox论文](https://link-to-paper) - 理论基础
- [API文档](man/) - 函数参考

## 🧪 测试

```r
# 运行稀疏效果测试
source("tests/test_sparse_effect.R")

# 运行完整集成测试
source("tests/test_integration.R")
```

## 🤝 贡献

欢迎提交Issue和Pull Request！

### 开发环境设置

1. Fork本仓库
2. 创建特性分支: `git checkout -b feature/your-feature`
3. 提交更改: `git commit -am 'Add some feature'`
4. 推送分支: `git push origin feature/your-feature`
5. 提交Pull Request

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
