# TransCox-Sparse API 参考

## 📋 函数列表

### 主要函数

- [`runTransCox_Sparse()`](#runtranscox_sparse) - 主要接口函数
- [`SelParam_By_BIC_Sparse()`](#selparam_by_bic_sparse) - 参数选择函数
- [`GetBIC()`](#getbic) - BIC计算函数

### 辅助函数

- [`TransCox_Sparse()`](#transcox_sparse) - 核心算法实现
- [`TransCox()`](#transcox) - 非稀疏版本

---

## 🔧 函数详细说明

### `runTransCox_Sparse()`

**描述**: TransCox-Sparse的主要接口函数，提供完整的稀疏迁移学习Cox回归分析。

**语法**:
```r
runTransCox_Sparse(
  primData,
  auxData,
  cov = c("X1", "X2"),
  statusvar = "status",
  lambda1 = NULL,
  lambda2 = NULL,
  lambda_beta = NULL,
  lambda1_vec = c(0.001, 0.01, 0.1, 1),
  lambda2_vec = c(0.001, 0.01, 0.1, 1),
  lambda_beta_vec = c(0.001, 0.01, 0.1, 1),
  auto_tune = TRUE,
  verbose = TRUE,
  parallel = FALSE,
  learning_rate = 0.004,
  nsteps = 200,
  tolerance = 1e-6,
  early_stopping = TRUE,
  adaptive_lr = TRUE
)
```

**参数**:

| 参数 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| `primData` | data.frame | ✓ | - | 目标域数据，包含生存时间、状态和协变量 |
| `auxData` | data.frame | ✓ | - | 源域数据，结构与primData相同 |
| `cov` | character | ✗ | c("X1", "X2") | 协变量名称向量 |
| `statusvar` | character | ✗ | "status" | 状态变量名（0=删失，1=事件） |
| `lambda1` | numeric | ✗ | NULL | eta的L1惩罚参数，NULL时自动选择 |
| `lambda2` | numeric | ✗ | NULL | xi的L1惩罚参数，NULL时自动选择 |
| `lambda_beta` | numeric | ✗ | NULL | beta_t的L1惩罚参数，NULL时自动选择 |
| `lambda1_vec` | numeric | ✗ | c(0.001, 0.01, 0.1, 1) | lambda1候选值向量 |
| `lambda2_vec` | numeric | ✗ | c(0.001, 0.01, 0.1, 1) | lambda2候选值向量 |
| `lambda_beta_vec` | numeric | ✗ | c(0.001, 0.01, 0.1, 1) | lambda_beta候选值向量 |
| `auto_tune` | logical | ✗ | TRUE | 是否启用自动参数调优 |
| `verbose` | logical | ✗ | TRUE | 是否显示详细信息 |
| `parallel` | logical | ✗ | FALSE | 是否启用并行计算 |
| `learning_rate` | numeric | ✗ | 0.004 | 优化算法学习率 |
| `nsteps` | integer | ✗ | 200 | 最大优化步数 |
| `tolerance` | numeric | ✗ | 1e-6 | 收敛容忍度 |
| `early_stopping` | logical | ✗ | TRUE | 是否启用早停机制 |
| `adaptive_lr` | logical | ✗ | TRUE | 是否使用自适应学习率 |

**返回值**:
```r
list(
  eta = numeric,              # 系数差异向量
  xi = numeric,               # 基线风险调整
  new_beta = numeric,         # 最终系数向量
  lambda1_used = numeric,     # 使用的lambda1值
  lambda2_used = numeric,     # 使用的lambda2值
  lambda_beta_used = numeric, # 使用的lambda_beta值
  nonzero_count = integer,    # 非零系数数量
  sparsity_ratio = numeric,   # 稀疏度比例
  convergence_info = list,    # 收敛信息
  computation_time = numeric  # 计算时间（秒）
)
```

**示例**:
```r
# 基本使用
result <- runTransCox_Sparse(
  primData = prim_data,
  auxData = aux_data,
  cov = c("X1", "X2", "X3")
)

# 自定义参数
result <- runTransCox_Sparse(
  primData = prim_data,
  auxData = aux_data,
  cov = feature_names,
  lambda1 = 0.01,
  lambda2 = 0.01,
  lambda_beta = 0.1,
  auto_tune = FALSE
)
```

---

### `SelParam_By_BIC_Sparse()`

**描述**: 使用BIC准则选择最优参数组合的函数。

**语法**:
```r
SelParam_By_BIC_Sparse(
  primData,
  auxData,
  cov,
  statusvar = "status",
  lambda1_vec,
  lambda2_vec,
  lambda_beta_vec,
  verbose = TRUE,
  parallel = FALSE,
  learning_rate = 0.004,
  nsteps = 200,
  tolerance = 1e-6
)
```

**参数**:

| 参数 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| `primData` | data.frame | ✓ | - | 目标域数据 |
| `auxData` | data.frame | ✓ | - | 源域数据 |
| `cov` | character | ✓ | - | 协变量名称向量 |
| `statusvar` | character | ✗ | "status" | 状态变量名 |
| `lambda1_vec` | numeric | ✓ | - | lambda1候选值向量 |
| `lambda2_vec` | numeric | ✓ | - | lambda2候选值向量 |
| `lambda_beta_vec` | numeric | ✓ | - | lambda_beta候选值向量 |
| `verbose` | logical | ✗ | TRUE | 是否显示详细信息 |
| `parallel` | logical | ✗ | FALSE | 是否启用并行计算 |
| `learning_rate` | numeric | ✗ | 0.004 | 学习率 |
| `nsteps` | integer | ✗ | 200 | 最大步数 |
| `tolerance` | numeric | ✗ | 1e-6 | 收敛容忍度 |

**返回值**:
```r
list(
  eta = numeric,              # 最优eta
  xi = numeric,               # 最优xi
  new_beta = numeric,         # 最优beta
  lambda1_optimal = numeric,  # 最优lambda1
  lambda2_optimal = numeric,  # 最优lambda2
  lambda_beta_optimal = numeric, # 最优lambda_beta
  min_bic = numeric,          # 最小BIC值
  bic_results = data.frame    # 所有参数组合的BIC结果
)
```

---

### `GetBIC()`

**描述**: 计算给定参数下的BIC值。

**语法**:
```r
GetBIC(primData, cov, statusvar, eta, xi, new_beta)
```

**参数**:

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `primData` | data.frame | ✓ | 目标域数据 |
| `cov` | character | ✓ | 协变量名称向量 |
| `statusvar` | character | ✓ | 状态变量名 |
| `eta` | numeric | ✓ | 系数差异向量 |
| `xi` | numeric | ✓ | 基线风险调整 |
| `new_beta` | numeric | ✓ | 最终系数向量 |

**返回值**:
```r
numeric  # BIC值
```

---

### `TransCox_Sparse()`

**描述**: 核心稀疏迁移学习算法实现（Python函数）。

**语法**:
```r
TransCox_Sparse(
  prim_time, prim_status, prim_X,
  aux_time, aux_status, aux_X,
  lambda1, lambda2, lambda_beta,
  learning_rate = 0.004,
  nsteps = 200,
  tolerance = 1e-6
)
```

**参数**:

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `prim_time` | numeric | ✓ | 目标域生存时间 |
| `prim_status` | numeric | ✓ | 目标域状态 |
| `prim_X` | matrix | ✓ | 目标域协变量矩阵 |
| `aux_time` | numeric | ✓ | 源域生存时间 |
| `aux_status` | numeric | ✓ | 源域状态 |
| `aux_X` | matrix | ✓ | 源域协变量矩阵 |
| `lambda1` | numeric | ✓ | eta的L1惩罚参数 |
| `lambda2` | numeric | ✓ | xi的L1惩罚参数 |
| `lambda_beta` | numeric | ✓ | beta_t的L1惩罚参数 |
| `learning_rate` | numeric | ✗ | 学习率 |
| `nsteps` | integer | ✗ | 最大步数 |
| `tolerance` | numeric | ✗ | 收敛容忍度 |

**返回值**:
```r
list(
  eta = numeric,      # 系数差异向量
  xi = numeric,       # 基线风险调整
  new_beta = numeric  # 最终系数向量
)
```

---

## 🎯 使用模式

### 1. 快速开始模式

```r
# 使用默认参数，自动调优
result <- runTransCox_Sparse(primData, auxData, cov = features)
```

### 2. 自定义参数模式

```r
# 指定特定参数
result <- runTransCox_Sparse(
  primData, auxData, cov = features,
  lambda1 = 0.01, lambda2 = 0.01, lambda_beta = 0.1,
  auto_tune = FALSE
)
```

### 3. 高性能模式

```r
# 启用并行计算和优化设置
result <- runTransCox_Sparse(
  primData, auxData, cov = features,
  parallel = TRUE,
  early_stopping = TRUE,
  adaptive_lr = TRUE
)
```

### 4. 调试模式

```r
# 详细输出和较少步数用于调试
result <- runTransCox_Sparse(
  primData, auxData, cov = features,
  verbose = TRUE,
  nsteps = 50,
  tolerance = 1e-4
)
```

---

## ⚠️ 注意事项

1. **数据格式**: 确保primData和auxData具有相同的列结构
2. **缺失值**: 算法不处理缺失值，请预先处理
3. **内存使用**: 大数据集建议启用early_stopping和适当的tolerance
4. **Python依赖**: 确保TensorFlow 2.10.0正确安装
5. **并行计算**: Windows系统下并行计算可能不稳定

---

## 🔗 相关文档

- [用户手册](user_manual.md)
- [性能优化指南](performance_guide.md)
- [示例代码](examples/)