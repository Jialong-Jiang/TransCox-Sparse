# TransCox-Sparse 用户手册

## 📖 概述

TransCox-Sparse是一个专为高维稀疏生存分析设计的R包，通过迁移学习技术从源域数据中学习知识来改善目标域的Cox回归模型性能。

## 🚀 快速开始

### 1. 环境准备

```r
# 安装必要的R包
install.packages(c("reticulate", "survival", "glmnet"))

# 配置Python环境
library(reticulate)
# 确保安装了TensorFlow 2.10.0
```

### 2. 基本使用

```r
# 加载包
library(TransCox)

# 准备数据
# primData: 目标域数据，包含time, status, 协变量
# auxData: 源域数据，结构与primData相同

# 运行稀疏TransCox
result <- runTransCox_Sparse(
    primData = prim_data,
    auxData = aux_data,
    cov = c("X1", "X2", "X3"),  # 协变量名称
    statusvar = "status",
    auto_tune = TRUE,           # 启用自动调参
    verbose = TRUE
)

# 查看结果
print(result)
```

## 📊 参数说明

### 主要参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `primData` | data.frame | - | 目标域数据 |
| `auxData` | data.frame | - | 源域数据 |
| `cov` | character | c("X1", "X2") | 协变量名称向量 |
| `statusvar` | character | "status" | 状态变量名 |
| `lambda1` | numeric | NULL | eta的L1惩罚参数 |
| `lambda2` | numeric | NULL | xi的L1惩罚参数 |
| `lambda_beta` | numeric | NULL | beta_t的L1惩罚参数 |
| `auto_tune` | logical | TRUE | 是否启用自动调参 |
| `verbose` | logical | TRUE | 是否显示详细信息 |
| `parallel` | logical | FALSE | 是否启用并行计算 |

### 高级参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `learning_rate` | numeric | 0.004 | 学习率 |
| `nsteps` | integer | 200 | 优化步数 |
| `tolerance` | numeric | 1e-6 | 收敛容忍度 |
| `early_stopping` | logical | TRUE | 是否启用早停 |
| `adaptive_lr` | logical | TRUE | 是否自适应学习率 |

## 🎯 使用场景

### 1. 高维稀疏数据

```r
# 当特征数 >> 样本数时
result <- runTransCox_Sparse(
    primData = prim_data,
    auxData = aux_data,
    cov = paste0("gene_", 1:1000),  # 1000个基因特征
    lambda_beta = 0.1,              # 强稀疏约束
    verbose = TRUE
)
```

### 2. 自动参数选择

```r
# 让算法自动选择最优参数
result <- runTransCox_Sparse(
    primData = prim_data,
    auxData = aux_data,
    cov = feature_names,
    lambda1 = NULL,     # 自动选择
    lambda2 = NULL,     # 自动选择
    lambda_beta = NULL, # 自动选择
    auto_tune = TRUE
)
```

### 3. 快速原型开发

```r
# 减少计算时间用于快速测试
result <- runTransCox_Sparse(
    primData = prim_data,
    auxData = aux_data,
    cov = feature_names,
    nsteps = 100,       # 减少步数
    verbose = FALSE,    # 关闭详细输出
    auto_tune = FALSE   # 使用默认参数
)
```

## 📈 结果解释

### 返回对象结构

```r
# result包含以下组件：
result$eta              # 系数差异向量
result$xi               # 基线风险调整
result$new_beta         # 最终系数 (estR + eta)
result$lambda1_used     # 使用的lambda1值
result$lambda2_used     # 使用的lambda2值
result$lambda_beta_used # 使用的lambda_beta值
result$nonzero_count    # 非零系数数量
result$sparsity_ratio   # 稀疏度比例
result$convergence_info # 收敛信息
```

### 模型评估

```r
# 提取最终系数
final_coef <- result$new_beta

# 计算非零系数
nonzero_features <- which(abs(final_coef) > 1e-8)
cat("选择的特征数:", length(nonzero_features), "\n")

# 稀疏度
sparsity <- 1 - length(nonzero_features) / length(final_coef)
cat("稀疏度:", round(sparsity * 100, 2), "%\n")
```

## ⚡ 性能优化

### 1. 分层搜索策略

算法自动使用分层搜索：
- **粗搜索**: 快速定位最优区域
- **细搜索**: 在最优区域精细调优

### 2. 早停机制

自动检测并跳过无效参数组合：
- 参数有效性检查
- 结果有效性验证
- BIC值合理性检查

### 3. 智能缓存

避免重复计算相似参数组合，显著提升性能。

## 🔧 故障排除

### 常见问题

1. **Python环境问题**
```r
# 检查Python配置
reticulate::py_config()

# 重新加载Python函数
reticulate::source_python("inst/python/TransCoxFunction_Sparse.py")
```

2. **内存不足**
```r
# 减少参数网格大小
result <- runTransCox_Sparse(
    ...,
    nsteps = 100,       # 减少步数
    parallel = FALSE    # 关闭并行
)
```

3. **收敛问题**
```r
# 调整学习率和步数
result <- runTransCox_Sparse(
    ...,
    learning_rate = 0.001,  # 降低学习率
    nsteps = 500,           # 增加步数
    tolerance = 1e-8        # 提高精度要求
)
```

## 📚 更多资源

- [API参考](api_reference.md)
- [性能优化指南](performance_guide.md)
- [示例代码](examples/)
- [性能优化总结](../PERFORMANCE_OPTIMIZATION_SUMMARY.md)

## 🤝 支持

如有问题，请查看：
1. 本用户手册
2. 函数文档 (`?runTransCox_Sparse`)
3. 示例代码
4. GitHub Issues