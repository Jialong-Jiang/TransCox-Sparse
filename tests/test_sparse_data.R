#' 高维稀疏TransCox测试脚本
#' 
#' 创建测试数据并验证修改后的代码功能

# 加载必要的包
library(survival)
library(Matrix)
library(glmnet)
library(reticulate)

# 设置工作目录
setwd("c:/Users/jiang/Desktop/cursor-cox/TransCox_Modified/TransCox")

# 加载所有必要的函数
source("R/GetAuxSurv.R")
source("R/GetPrimaryParam.R")
source("R/deltaQ.R")  # 添加deltaQ函数
source("R/GetAuxSurv_Sparse.R")
source("R/SelParam_By_BIC_Sparse.R")
source("R/runTransCox_Sparse.R")

# 创建高维稀疏测试数据生成函数
generate_sparse_survival_data <- function(n, p, sparsity = 0.9, 
                                         true_beta_sparsity = 0.8,
                                         seed = 123) {
    set.seed(seed)
    
    cat("生成高维稀疏生存数据:\n")
    cat("  样本数 n =", n, "\n")
    cat("  特征数 p =", p, "\n")
    cat("  协变量稀疏度 =", sparsity, "\n")
    cat("  真实系数稀疏度 =", true_beta_sparsity, "\n")
    
    # 生成稀疏协变量矩阵
    # 使用块对角结构模拟基因组数据的相关性
    X <- matrix(0, n, p)
    
    # 生成一些相关的特征块
    block_size <- min(10, p %/% 5)
    n_blocks <- p %/% block_size
    
    for (i in 1:n_blocks) {
        start_idx <- (i-1) * block_size + 1
        end_idx <- min(i * block_size, p)
        block_p <- end_idx - start_idx + 1
        
        # 生成相关的正态变量
        Sigma <- 0.3^abs(outer(1:block_p, 1:block_p, "-"))
        X_block <- MASS::mvrnorm(n, rep(0, block_p), Sigma)
        
        # 引入稀疏性
        sparse_mask <- matrix(rbinom(n * block_p, 1, 1 - sparsity), n, block_p)
        X[, start_idx:end_idx] <- X_block * sparse_mask
    }
    
    # 处理剩余的特征
    if (p %% block_size != 0) {
        remaining_start <- n_blocks * block_size + 1
        remaining_features <- p - remaining_start + 1
        X[, remaining_start:p] <- matrix(rnorm(n * remaining_features), n, remaining_features) * 
                                 matrix(rbinom(n * remaining_features, 1, 1 - sparsity), n, remaining_features)
    }
    
    # 生成真实的稀疏系数
    true_beta <- rep(0, p)
    n_nonzero <- round(p * (1 - true_beta_sparsity))
    nonzero_idx <- sample(1:p, n_nonzero)
    
    # 设置一些强效应和一些弱效应
    strong_effects <- sample(nonzero_idx, n_nonzero %/% 3)
    weak_effects <- setdiff(nonzero_idx, strong_effects)
    
    true_beta[strong_effects] <- rnorm(length(strong_effects), 0, 0.8)
    true_beta[weak_effects] <- rnorm(length(weak_effects), 0, 0.3)
    
    # 生成线性预测子
    linear_pred <- as.vector(X %*% true_beta)
    
    # 生成生存时间
    # 使用指数分布
    lambda_base <- 0.1
    hazard <- lambda_base * exp(linear_pred)
    time <- rexp(n, hazard)
    
    # 生成删失时间
    censor_time <- rexp(n, 0.02)  # 更低的删失率，避免过多删失
    
    # 观察时间和状态
    obs_time <- pmin(time, censor_time)
    status <- ifelse(time <= censor_time, 2, 1)  # 2=事件，1=删失
    
    # 创建数据框
    cov_names <- paste0("X", 1:p)
    colnames(X) <- cov_names
    
    data <- data.frame(
        time = obs_time,
        status = status,
        X
    )
    
    # 转换为稀疏矩阵格式
    X_sparse <- Matrix(X, sparse = TRUE)
    
    cat("数据生成完成:\n")
    cat("  事件率:", round(mean(status), 3), "\n")
    cat("  协变量矩阵稀疏度:", round(1 - nnzero(X_sparse) / (n * p), 3), "\n")
    cat("  真实系数非零个数:", sum(abs(true_beta) > 1e-8), "\n")
    
    return(list(
        data = data,
        X_sparse = X_sparse,
        true_beta = true_beta,
        cov_names = cov_names,
        nonzero_idx = nonzero_idx
    ))
}

# 测试1: 中等维度数据 (p < n)
cat("\n=== 测试1: 中等维度数据 ===\n")
test1_data <- generate_sparse_survival_data(n = 200, p = 50, sparsity = 0.7, seed = 123)

# 分割为源域和目标域
n_aux <- 120
n_prim <- 80

aux_idx <- 1:n_aux
prim_idx <- (n_aux + 1):(n_aux + n_prim)

auxData1 <- test1_data$data[aux_idx, ]
primData1 <- test1_data$data[prim_idx, ]

cat("源域数据:", nrow(auxData1), "样本\n")
cat("目标域数据:", nrow(primData1), "样本\n")

# 测试2: 高维数据 (p >> n)
cat("\n=== 测试2: 高维数据 ===\n")
test2_data <- generate_sparse_survival_data(n = 100, p = 500, sparsity = 0.95, seed = 456)

# 分割数据
n_aux2 <- 60
n_prim2 <- 40

aux_idx2 <- 1:n_aux2
prim_idx2 <- (n_aux2 + 1):(n_aux2 + n_prim2)

auxData2 <- test2_data$data[aux_idx2, ]
primData2 <- test2_data$data[prim_idx2, ]

cat("源域数据:", nrow(auxData2), "样本\n")
cat("目标域数据:", nrow(primData2), "样本\n")

# 保存测试数据
save(test1_data, auxData1, primData1, 
     test2_data, auxData2, primData2,
     file = "tests/sparse_test_data.RData")

cat("\n测试数据已保存到 tests/sparse_test_data.RData\n")

# 简单功能测试
cat("\n=== 简单功能测试 ===\n")

# 测试稀疏版本的GetAuxSurv
cat("测试GetAuxSurv_Sparse...\n")
cov_names1 <- test1_data$cov_names

tryCatch({
    aux_result1 <- GetAuxSurv_Sparse(auxData1, cov = cov_names1)
    cat("✓ GetAuxSurv_Sparse 成功\n")
    cat("  估计的系数数:", length(aux_result1$estR), "\n")
    cat("  非零系数数:", sum(abs(aux_result1$estR) > 1e-6), "\n")
}, error = function(e) {
    cat("✗ GetAuxSurv_Sparse 失败:", e$message, "\n")
})

# 测试GetPrimaryParam
cat("测试GetPrimaryParam...\n")
tryCatch({
    prim_result1 <- GetPrimaryParam(primData1, q = aux_result1$q, estR = aux_result1$estR)
    cat("✓ GetPrimaryParam 成功\n")
}, error = function(e) {
    cat("✗ GetPrimaryParam 失败:", e$message, "\n")
})

cat("\n基础功能测试完成！\n")
cat("现在可以运行完整的TransCox分析。\n")