# TransCox 用户手册

## 目录
1. [包概述](#包概述)
2. [原始TransCox模块说明](#原始transcox模块说明)
3. [稀疏版本功能](#稀疏版本功能)
4. [安装和环境配置](#安装和环境配置)
5. [使用指南](#使用指南)
6. [超参数调整](#超参数调整)
7. [示例和最佳实践](#示例和最佳实践)

---

## 包概述

TransCox是一个用于生存分析的迁移学习R包，能够利用源域（辅助）数据来改善目标域（主要）数据的Cox回归模型性能。本包现已扩展支持高维稀疏数据分析。

### 主要特性
- **迁移学习**: 从源域数据中学习知识来改善目标域模型
- **高维数据支持**: 处理特征数远大于样本数的情况（p >> n）
- **稀疏性**: 通过L1正则化实现特征选择
- **自动调参**: 基于BIC的超参数自动选择
- **向后兼容**: 完全兼容原始TransCox功能

---

## 原始TransCox模块说明

### 核心模块架构

#### 1. 数据预处理模块
- **`GenSimData.R`**: 生成模拟数据
  - 创建目标域和源域的生存数据
  - 支持不同的数据生成设置
  - 用于测试和验证

- **`GetAuxSurv.R`**: 源域参数估计
  - 从源域数据估计生存函数
  - 计算基线风险函数
  - 输出: `q`（累积风险）和 `estR`（回归系数）

- **`GetPrimaryParam.R`**: 目标域参数估计
  - 处理目标域数据
  - 计算累积风险函数
  - 准备优化所需的数据结构

#### 2. 核心算法模块
- **`runTransCox_one.R`**: 单次TransCox运行
  - 调用Python优化函数
  - 执行迁移学习算法
  - 返回eta（系数变化）和xi（风险变化）

- **`TransCoxFunction.py`**: Python优化引擎
  - 基于TensorFlow的梯度优化
  - L1正则化实现
  - 数值稳定性保证

#### 3. 参数选择模块
- **`SelLR_By_BIC.R`**: 学习率和步数选择
  - 基于BIC准则选择最优学习率
  - 确定最优优化步数
  - 网格搜索实现

- **`SelParam_By_BIC.R`**: 正则化参数选择
  - 选择最优lambda1（eta正则化）
  - 选择最优lambda2（xi正则化）
  - 交叉验证支持

#### 4. 辅助功能模块
- **`GetBIC.R`**: BIC计算
  - 模型选择准则
  - 考虑模型复杂度和拟合优度

- **`GetLogLike.R`**: 对数似然计算
  - Cox模型似然函数
  - 用于模型评估

- **`deltaQ.R`** & **`dQtocumQ.R`**: 风险函数转换
  - 离散风险到累积风险的转换
  - 数据格式标准化

### 模块间关联关系

```
数据输入 → GetAuxSurv → GetPrimaryParam → runTransCox_one → 结果输出
    ↓           ↓              ↓              ↑
GenSimData  SelParam_By_BIC  GetBIC    TransCoxFunction.py
                ↓              ↓
            SelLR_By_BIC   GetLogLike
```

---

## 稀疏版本功能

### 新增模块

#### 1. 稀疏数据处理
- **`GetAuxSurv_Sparse.R`**: 稀疏源域参数估计
  - 支持高维数据的源域分析
  - 优化内存使用
  - 数值稳定性增强

- **`runTransCox_Sparse.R`**: 稀疏TransCox主函数
  - 高维数据的完整分析流程
  - 自动稀疏性检测
  - 集成参数调优

#### 2. 稀疏优化引擎
- **`TransCoxFunction_Sparse.py`**: 稀疏优化实现
  - 支持beta_t的L1正则化
  - 软阈值化（Soft Thresholding）
  - 三重正则化：eta、xi、beta_t

#### 3. 稀疏参数选择
- **`SelParam_By_BIC_Sparse.R`**: 稀疏参数选择
  - 三维参数网格搜索
  - lambda_beta参数优化
  - 稀疏性控制

### 稀疏性实现原理

1. **L1正则化**: 对beta_t系数施加L1惩罚
2. **软阈值化**: 在每次梯度更新后应用软阈值操作
3. **自适应阈值**: 根据lambda_beta动态调整阈值
4. **稀疏性监控**: 实时监控非零系数比例

### 与原版本的兼容性

- **完全向后兼容**: 当`lambda_beta = 0`时，行为与原版本完全一致
- **输出格式一致**: 返回相同的结果结构
- **API兼容**: 支持原有的所有参数和功能

---

## 安装和环境配置

### 1. Python环境配置

```bash
# 创建conda环境
conda create -n TransCoxEnvi python=3.8
conda activate TransCoxEnvi

# 安装必要包
conda install tensorflow=2.10.0
conda install numpy pandas
```

### 2. R环境配置

```r
# 安装必要的R包
install.packages(c("survival", "glmnet", "Matrix", "reticulate"))

# 配置Python环境
library(reticulate)
use_condaenv("TransCoxEnvi")
use_python("path/to/your/conda/envs/TransCoxEnvi/python")

# 验证环境
tf <- import("tensorflow")
print(tf$`__version__`)
```

### 3. 加载TransCox

```r
# 设置工作目录
setwd("path/to/TransCox")

# 加载核心函数
source("R/GetAuxSurv.R")
source("R/GetPrimaryParam.R")
source("R/runTransCox_one.R")

# 加载稀疏版本函数
source("R/GetAuxSurv_Sparse.R")
source("R/runTransCox_Sparse.R")
source("R/SelParam_By_BIC_Sparse.R")

# 加载Python函数
source_python("inst/python/TransCoxFunction.py")
source_python("inst/python/TransCoxFunction_Sparse.py")
```

---

## 使用指南

### 1. 标准TransCox使用

```r
# 生成或准备数据
onedata <- GenSimData(nprim = 200, naux = 500, setting = 1)
primData <- onedata$primData
auxData <- onedata$auxData

# 基本分析流程
Cout <- GetAuxSurv(auxData, cov = c("X1", "X2"))
Pout <- GetPrimaryParam(primData, q = Cout$q, estR = Cout$estR)

# 运行TransCox
result <- runTransCox_one(Pout, l1 = 0.1, l2 = 0.1, 
                         learning_rate = 0.004, nsteps = 200)

# 查看结果
print(result$new_beta)  # 新的回归系数
```

### 2. 稀疏TransCox使用

#### 基本使用
```r
# 高维稀疏数据分析
result_sparse <- runTransCox_Sparse(
    primData = primData,
    auxData = auxData,
    cov = paste0("X", 1:50),  # 50个特征
    statusvar = "status",
    lambda1 = 0.1,
    lambda2 = 0.1,
    lambda_beta = 0.05,  # 稀疏性控制参数
    learning_rate = 0.004,
    nsteps = 200,
    use_sparse = TRUE,
    verbose = TRUE
)

# 查看稀疏性结果
cat("非零系数数:", sum(abs(result_sparse$new_beta) > 1e-8), "\n")
cat("稀疏度:", round((1 - sum(abs(result_sparse$new_beta) > 1e-8) / 
                    length(result_sparse$new_beta)) * 100, 2), "%\n")
```

#### 自动参数调优
```r
# 自动选择最优参数
result_auto <- runTransCox_Sparse(
    primData = primData,
    auxData = auxData,
    cov = paste0("X", 1:100),
    statusvar = "status",
    lambda1 = NULL,      # 自动选择
    lambda2 = NULL,      # 自动选择
    lambda_beta = NULL,  # 自动选择
    auto_tune = TRUE,
    verbose = TRUE
)
```

### 3. 手动参数调优

```r
# 定义参数网格
lambda1_vec <- c(0.01, 0.05, 0.1, 0.2)
lambda2_vec <- c(0.01, 0.05, 0.1, 0.2)
lambda_beta_vec <- c(0, 0.01, 0.05, 0.1, 0.2)

# BIC参数选择
bic_result <- SelParam_By_BIC_Sparse(
    primData = primData,
    auxData = auxData,
    cov = paste0("X", 1:30),
    statusvar = "status",
    lambda1_vec = lambda1_vec,
    lambda2_vec = lambda2_vec,
    lambda_beta_vec = lambda_beta_vec,
    learning_rate = 0.004,
    nsteps = 100,
    verbose = TRUE
)

# 查看最优参数
cat("最优lambda1:", bic_result$best_lambda1, "\n")
cat("最优lambda2:", bic_result$best_lambda2, "\n")
cat("最优lambda_beta:", bic_result$best_lambda_beta, "\n")
```

---

## 超参数调整

### 1. 关键参数说明

#### 正则化参数
- **`lambda1`**: eta（系数变化）的L1惩罚强度
  - 范围: 0.001 - 1.0
  - 较大值 → 更保守的系数迁移
  
- **`lambda2`**: xi（风险变化）的L1惩罚强度
  - 范围: 0.001 - 1.0
  - 较大值 → 更保守的风险迁移

- **`lambda_beta`**: beta_t（最终系数）的L1惩罚强度
  - 范围: 0 - 0.5
  - 0 → 无稀疏性（标准TransCox）
  - 0.01-0.05 → 轻度稀疏
  - 0.1-0.2 → 强稀疏
  - >0.2 → 可能过度稀疏

#### 优化参数
- **`learning_rate`**: 学习率
  - 推荐: 0.001 - 0.01
  - 高维数据建议使用较小值（0.001-0.004）

- **`nsteps`**: 优化步数
  - 低维: 100-200步
  - 高维: 200-500步

### 2. 针对不同数据类型的参数建议

#### 低维数据 (p < 50)
```r
# 推荐参数
lambda1 = 0.1
lambda2 = 0.1
lambda_beta = 0.01  # 轻度稀疏
learning_rate = 0.004
nsteps = 200
```

#### 中维数据 (50 ≤ p ≤ 500)
```r
# 推荐参数
lambda1 = 0.05
lambda2 = 0.05
lambda_beta = 0.05  # 中度稀疏
learning_rate = 0.002
nsteps = 300
```

#### 高维数据 (p > 500)
```r
# 推荐参数
lambda1 = 0.01
lambda2 = 0.01
lambda_beta = 0.1   # 强稀疏
learning_rate = 0.001
nsteps = 500
```

### 3. 稀疏性调整策略

#### 根据期望稀疏度选择lambda_beta
- **0% 稀疏度**: `lambda_beta = 0`
- **10-30% 稀疏度**: `lambda_beta = 0.01-0.03`
- **30-60% 稀疏度**: `lambda_beta = 0.05-0.08`
- **60-90% 稀疏度**: `lambda_beta = 0.1-0.15`
- **>90% 稀疏度**: `lambda_beta = 0.2+`

#### 自适应调整
```r
# 从小参数开始，逐步增加
lambda_beta_sequence <- c(0, 0.01, 0.05, 0.1, 0.2)

for(lb in lambda_beta_sequence) {
    result <- runTransCox_Sparse(
        primData = primData,
        auxData = auxData,
        cov = cov_names,
        lambda_beta = lb,
        verbose = FALSE
    )
    
    sparsity <- (1 - sum(abs(result$new_beta) > 1e-8) / 
                length(result$new_beta)) * 100
    
    cat("lambda_beta =", lb, ", 稀疏度 =", round(sparsity, 1), "%\n")
    
    # 如果达到期望稀疏度，停止
    if(sparsity >= target_sparsity) break
}
```

---

## 示例和最佳实践

### 1. 完整分析流程示例

```r
# === 新数据分析完整流程 ===

# 1. 环境准备
library(survival)
library(Matrix)
library(reticulate)
use_condaenv("TransCoxEnvi")

# 加载TransCox
setwd("path/to/TransCox")
source("R/runTransCox_Sparse.R")
source_python("inst/python/TransCoxFunction_Sparse.py")

# 2. 数据准备
# 假设您有：
# - primData: 目标域数据（包含time, status, X1, X2, ..., Xp）
# - auxData: 源域数据（相同格式）
# - 高维特征: X1, X2, ..., X500

cov_names <- paste0("X", 1:500)  # 500个特征

# 3. 数据质量检查
cat("目标域样本数:", nrow(primData), "\n")
cat("源域样本数:", nrow(auxData), "\n")
cat("特征数:", length(cov_names), "\n")
cat("事件率:", mean(primData$status), "\n")

# 4. 稀疏性分析
# 首先运行无稀疏版本作为基线
baseline_result <- runTransCox_Sparse(
    primData = primData,
    auxData = auxData,
    cov = cov_names,
    statusvar = "status",
    lambda_beta = 0,  # 无稀疏
    auto_tune = TRUE,
    verbose = TRUE
)

# 5. 稀疏版本分析
sparse_result <- runTransCox_Sparse(
    primData = primData,
    auxData = auxData,
    cov = cov_names,
    statusvar = "status",
    lambda_beta = NULL,  # 自动选择
    auto_tune = TRUE,
    use_sparse = TRUE,
    verbose = TRUE
)

# 6. 结果比较
cat("\n=== 结果比较 ===\n")
cat("基线版本非零系数:", sum(abs(baseline_result$new_beta) > 1e-8), "\n")
cat("稀疏版本非零系数:", sum(abs(sparse_result$new_beta) > 1e-8), "\n")

sparsity <- (1 - sum(abs(sparse_result$new_beta) > 1e-8) / 
            length(sparse_result$new_beta)) * 100
cat("稀疏度:", round(sparsity, 1), "%\n")

# 7. 特征重要性分析
important_features <- which(abs(sparse_result$new_beta) > 1e-8)
cat("重要特征:", paste(cov_names[important_features], collapse = ", "), "\n")
```

### 2. 参数敏感性分析

```r
# 测试不同lambda_beta值的效果
lambda_beta_values <- c(0, 0.01, 0.05, 0.1, 0.2)
results_summary <- data.frame(
    lambda_beta = lambda_beta_values,
    nonzero_coef = NA,
    sparsity = NA,
    bic = NA
)

for(i in seq_along(lambda_beta_values)) {
    result <- runTransCox_Sparse(
        primData = primData,
        auxData = auxData,
        cov = cov_names,
        lambda_beta = lambda_beta_values[i],
        verbose = FALSE
    )
    
    nonzero <- sum(abs(result$new_beta) > 1e-8)
    sparsity <- (1 - nonzero / length(result$new_beta)) * 100
    
    results_summary$nonzero_coef[i] <- nonzero
    results_summary$sparsity[i] <- sparsity
    
    cat("lambda_beta =", lambda_beta_values[i], 
        ", 非零系数 =", nonzero,
        ", 稀疏度 =", round(sparsity, 1), "%\n")
}

print(results_summary)
```

### 3. 模型验证

```r
# 交叉验证评估
set.seed(123)
n_folds <- 5
fold_indices <- sample(rep(1:n_folds, length.out = nrow(primData)))

cv_results <- list()

for(fold in 1:n_folds) {
    train_idx <- which(fold_indices != fold)
    test_idx <- which(fold_indices == fold)
    
    train_data <- primData[train_idx, ]
    test_data <- primData[test_idx, ]
    
    # 训练模型
    model <- runTransCox_Sparse(
        primData = train_data,
        auxData = auxData,
        cov = cov_names,
        lambda_beta = 0.05,  # 使用之前确定的最优参数
        verbose = FALSE
    )
    
    # 预测（这里需要实现预测函数）
    # predictions <- predict_transcox(model, test_data)
    # cv_results[[fold]] <- evaluate_predictions(predictions, test_data)
}
```

### 4. 最佳实践建议

#### 数据预处理
1. **标准化特征**: 确保所有特征在相似尺度
2. **缺失值处理**: 使用适当的插补方法
3. **异常值检测**: 识别和处理极端值

#### 参数选择
1. **从粗网格开始**: 先用粗糙的参数网格快速筛选
2. **精细调优**: 在最优区域进行精细搜索
3. **交叉验证**: 使用交叉验证避免过拟合

#### 结果解释
1. **稀疏性合理性**: 确保稀疏度符合领域知识
2. **特征重要性**: 验证选中特征的生物学意义
3. **模型稳定性**: 检查结果在不同随机种子下的稳定性

---

## 故障排除

### 常见问题及解决方案

1. **Python环境问题**
   ```r
   # 检查Python环境
   py_config()
   
   # 重新配置
   use_condaenv("TransCoxEnvi", required = TRUE)
   ```

2. **内存不足**
   ```r
   # 减少特征数或样本数
   # 增加系统内存
   # 使用分批处理
   ```

3. **收敛问题**
   ```r
   # 减小学习率
   # 增加优化步数
   # 检查数据质量
   ```

4. **稀疏性过度**
   ```r
   # 减小lambda_beta
   # 检查数据预处理
   # 验证参数设置
   ```

---

## 更新日志

### v2.0 (稀疏版本)
- 新增高维稀疏数据支持
- 实现L1正则化和软阈值化
- 添加自动参数调优功能
- 完全向后兼容

### v1.0 (原始版本)
- 基础TransCox迁移学习功能
- 低维数据支持
- BIC参数选择

---

## 联系和支持

如有问题或建议，请联系开发团队或查阅相关文档。

---

*本手册最后更新: 2024年*