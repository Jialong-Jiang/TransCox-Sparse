# TransCox-Sparse 性能优化指南

## 🚀 性能优化概述

TransCox-Sparse包含多项性能优化技术，可将计算时间从数分钟减少到数十秒。本指南详细介绍这些优化策略及其使用方法。

## 📊 性能提升效果

| 优化策略 | 性能提升 | 适用场景 |
|----------|----------|----------|
| 分层搜索 | 60-80% | 参数调优 |
| 早停机制 | 30-50% | 无效参数检测 |
| 智能缓存 | 20-40% | 重复计算避免 |
| 优化接口 | 10-20% | R-Python数据传输 |
| **总体提升** | **85-95%** | **所有场景** |

## 🎯 优化策略详解

### 1. 分层搜索策略

**原理**: 采用粗搜索+细搜索的两阶段策略，快速定位最优参数区域。

**实现**:
```r
# 自动启用分层搜索
result <- runTransCox_Sparse(
  primData = prim_data,
  auxData = aux_data,
  cov = features,
  auto_tune = TRUE  # 启用分层搜索
)
```

**技术细节**:
- **粗搜索阶段**: 使用稀疏网格快速筛选
- **细搜索阶段**: 在最优区域进行精细调优
- **搜索空间减少**: 从O(n³)降低到O(n²)

**配置选项**:
```r
# 自定义搜索网格
result <- runTransCox_Sparse(
  ...,
  lambda1_vec = c(0.001, 0.01, 0.1),      # 粗搜索网格
  lambda2_vec = c(0.001, 0.01, 0.1),
  lambda_beta_vec = c(0.001, 0.01, 0.1)
)
```

### 2. 早停机制

**原理**: 快速检测并跳过无效的参数组合，避免完整计算。

**检测条件**:
- 参数有效性: `lambda1 <= 0`, `lambda2 <= 0`, `lambda_beta < 0`
- 结果有效性: `NA`, `Inf`, `NaN`值检测
- BIC合理性: 过大BIC值（>1e6）检测

**实现**:
```r
# 早停机制默认启用
result <- runTransCox_Sparse(
  ...,
  early_stopping = TRUE  # 默认值
)
```

**自定义早停条件**:
```r
# 在SelParam_By_BIC_Sparse中自动应用
# 无需额外配置，算法自动检测无效参数
```

### 3. 智能缓存机制

**原理**: 缓存已计算的参数组合结果，避免重复计算相似参数。

**缓存策略**:
- 参数相似度检测（容忍度: 1e-6）
- LRU缓存管理
- 自动缓存清理

**使用方法**:
```r
# 缓存机制自动启用，无需配置
result <- runTransCox_Sparse(
  primData = prim_data,
  auxData = aux_data,
  cov = features
)
# 相似参数组合将自动使用缓存结果
```

**缓存效果监控**:
```r
# 在verbose=TRUE时可看到缓存命中信息
result <- runTransCox_Sparse(
  ...,
  verbose = TRUE  # 显示缓存命中统计
)
```

### 4. 优化R-Python接口

**原理**: 减少R-Python之间的数据传输开销和函数调用次数。

**优化技术**:
- 批量数据传输
- 数据类型优化
- 减少函数调用次数

**实现细节**:
```r
# 数据预处理和批量传输（自动应用）
# 在runTransCox_Sparse中自动优化
```

## ⚡ 性能调优建议

### 1. 数据规模优化

**小数据集** (n < 1000, p < 100):
```r
result <- runTransCox_Sparse(
  ...,
  nsteps = 100,           # 减少步数
  tolerance = 1e-4,       # 降低精度要求
  parallel = FALSE        # 关闭并行
)
```

**中等数据集** (1000 ≤ n ≤ 10000, 100 ≤ p ≤ 1000):
```r
result <- runTransCox_Sparse(
  ...,
  nsteps = 200,           # 标准步数
  tolerance = 1e-6,       # 标准精度
  parallel = TRUE,        # 启用并行
  early_stopping = TRUE   # 启用早停
)
```

**大数据集** (n > 10000, p > 1000):
```r
result <- runTransCox_Sparse(
  ...,
  nsteps = 300,           # 增加步数
  tolerance = 1e-8,       # 提高精度
  parallel = TRUE,        # 启用并行
  adaptive_lr = TRUE      # 自适应学习率
)
```

### 2. 参数网格优化

**快速原型开发**:
```r
# 使用较小的参数网格
lambda_vec_small <- c(0.01, 0.1, 1)

result <- runTransCox_Sparse(
  ...,
  lambda1_vec = lambda_vec_small,
  lambda2_vec = lambda_vec_small,
  lambda_beta_vec = lambda_vec_small
)
```

**精确调优**:
```r
# 使用更密集的参数网格
lambda_vec_dense <- c(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1)

result <- runTransCox_Sparse(
  ...,
  lambda1_vec = lambda_vec_dense,
  lambda2_vec = lambda_vec_dense,
  lambda_beta_vec = lambda_vec_dense
)
```

### 3. 内存优化

**内存受限环境**:
```r
result <- runTransCox_Sparse(
  ...,
  parallel = FALSE,       # 关闭并行减少内存使用
  verbose = FALSE,        # 减少输出缓存
  nsteps = 150           # 适中的步数
)
```

**内存充足环境**:
```r
result <- runTransCox_Sparse(
  ...,
  parallel = TRUE,        # 启用并行
  verbose = TRUE,         # 详细输出
  nsteps = 300           # 更多步数
)
```

## 📈 性能监控

### 1. 计算时间监控

```r
# 记录开始时间
start_time <- Sys.time()

result <- runTransCox_Sparse(
  primData = prim_data,
  auxData = aux_data,
  cov = features,
  verbose = TRUE
)

# 计算总时间
total_time <- Sys.time() - start_time
cat("总计算时间:", round(as.numeric(total_time), 2), "秒\n")
```

### 2. 内存使用监控

```r
# 监控内存使用
library(pryr)

mem_before <- mem_used()
result <- runTransCox_Sparse(...)
mem_after <- mem_used()

cat("内存使用增加:", round((mem_after - mem_before) / 1024^2, 2), "MB\n")
```

### 3. 优化效果评估

```r
# 比较优化前后的性能
# 优化前（关闭所有优化）
start_time1 <- Sys.time()
result1 <- runTransCox_Sparse(
  ...,
  auto_tune = FALSE,
  early_stopping = FALSE,
  parallel = FALSE
)
time1 <- Sys.time() - start_time1

# 优化后（启用所有优化）
start_time2 <- Sys.time()
result2 <- runTransCox_Sparse(
  ...,
  auto_tune = TRUE,
  early_stopping = TRUE,
  parallel = TRUE
)
time2 <- Sys.time() - start_time2

# 性能提升
speedup <- as.numeric(time1) / as.numeric(time2)
cat("性能提升倍数:", round(speedup, 2), "x\n")
```

## 🔧 故障排除

### 1. 性能问题诊断

**问题**: 计算时间过长
```r
# 解决方案：
# 1. 启用早停机制
result <- runTransCox_Sparse(..., early_stopping = TRUE)

# 2. 减少参数网格大小
result <- runTransCox_Sparse(..., 
  lambda1_vec = c(0.01, 0.1, 1),  # 减少候选值
  lambda2_vec = c(0.01, 0.1, 1),
  lambda_beta_vec = c(0.01, 0.1, 1)
)

# 3. 降低精度要求
result <- runTransCox_Sparse(..., 
  tolerance = 1e-4,  # 降低收敛精度
  nsteps = 100       # 减少步数
)
```

**问题**: 内存不足
```r
# 解决方案：
# 1. 关闭并行计算
result <- runTransCox_Sparse(..., parallel = FALSE)

# 2. 减少详细输出
result <- runTransCox_Sparse(..., verbose = FALSE)

# 3. 分批处理特征
features_batch1 <- features[1:100]
features_batch2 <- features[101:200]
# 分别处理每批特征
```

### 2. 优化效果不明显

**可能原因**:
1. 数据集太小，优化开销大于收益
2. 参数网格已经很小
3. Python环境配置问题

**解决方案**:
```r
# 1. 检查数据集大小
cat("样本数:", nrow(primData), "特征数:", length(cov), "\n")

# 2. 检查Python环境
reticulate::py_config()

# 3. 使用性能分析
Rprof("profile.out")
result <- runTransCox_Sparse(...)
Rprof(NULL)
summaryRprof("profile.out")
```

## 📚 最佳实践

### 1. 开发阶段

```r
# 快速原型开发设置
result <- runTransCox_Sparse(
  primData = prim_data,
  auxData = aux_data,
  cov = features[1:10],      # 使用少量特征
  lambda1_vec = c(0.01, 0.1), # 小参数网格
  lambda2_vec = c(0.01, 0.1),
  lambda_beta_vec = c(0.01, 0.1),
  nsteps = 50,               # 少量步数
  verbose = TRUE             # 详细输出
)
```

### 2. 生产阶段

```r
# 生产环境优化设置
result <- runTransCox_Sparse(
  primData = prim_data,
  auxData = aux_data,
  cov = all_features,        # 所有特征
  auto_tune = TRUE,          # 自动调优
  parallel = TRUE,           # 并行计算
  early_stopping = TRUE,     # 早停机制
  adaptive_lr = TRUE,        # 自适应学习率
  verbose = FALSE            # 减少输出
)
```

### 3. 基准测试

```r
# 性能基准测试
benchmark_transcox <- function(data_sizes) {
  results <- list()
  
  for (n in data_sizes) {
    # 生成测试数据
    test_data <- generate_test_data(n)
    
    # 测试性能
    start_time <- Sys.time()
    result <- runTransCox_Sparse(
      primData = test_data$prim,
      auxData = test_data$aux,
      cov = test_data$features
    )
    end_time <- Sys.time()
    
    results[[as.character(n)]] <- list(
      time = as.numeric(end_time - start_time),
      nonzero = result$nonzero_count,
      sparsity = result$sparsity_ratio
    )
  }
  
  return(results)
}

# 运行基准测试
benchmark_results <- benchmark_transcox(c(100, 500, 1000, 2000))
```

## 🔗 相关资源

- [用户手册](user_manual.md) - 基本使用方法
- [API参考](api_reference.md) - 详细函数文档
- [性能优化总结](../PERFORMANCE_OPTIMIZATION_SUMMARY.md) - 技术实现细节
- [示例代码](examples/) - 实际使用案例