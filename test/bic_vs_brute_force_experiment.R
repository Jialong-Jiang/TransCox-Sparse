# BIC参数选择 vs 暴力搜索对比实验
# 使用与demo_transcox_auto_weibull.R完全相同的参数设置

suppressMessages({
  library(survival)
  library(reticulate)
  # 加载R端函数
  source("R/cox_lasso_model.R")
  source("setup_environment.R")
  source("R/runTransCox_Sparse.R")
  source("R/SelParam_By_BIC_Sparse.R")
  source("R/generate_sparse_survival_data.R")
  # 加载Python端函数（确保TensorFlow可用）
  try({
    reticulate::source_python("inst/python/TransCoxFunction_Sparse.py")
    reticulate::source_python("inst/python/TransCoxFunction.py")
  }, silent = TRUE)
})

cat("=== BIC参数选择 vs 暴力搜索对比实验 ===\n\n")

# 第一步: 使用与demo_transcox_auto_weibull.R完全相同的数据生成设置
cat("第一步: 生成数据（与auto_weibull完全相同设置）\n")

# 数据参数（与auto_weibull完全一致）
n_prim <- 120; n_aux <- 1000; n_test <- 300; p <- 100; n_active <- 15
active_indices <- sort(sample(1:p, n_active))
true_beta <- rep(0, p)
true_beta[active_indices] <- rnorm(n_active, mean = 0.9, sd = 0.3)

# 生成数据（Weibull），增强活跃特征相关性
cat("生成Weibull高维稀疏数据...\n")
sparse_data <- generate_sparse_survival_data(
  n_main = n_prim, n_aux = n_aux, n_test = n_test,
  p = p, p_active = n_active,
  beta_true = true_beta,
  transfer_strength = 0.95, noise_level = 0.06,
  censoring_rate = 0.25, seed = 123, verbose = TRUE
)

prim_data <- sparse_data$main_data
aux_data  <- sparse_data$aux_data
test_data <- sparse_data$test_data
true_beta <- as.vector(sparse_data$beta_true)
feature_names <- paste0("X", 1:p)

cat("数据生成完成:\n")
cat("- 主数据集:", nrow(prim_data), "样本,", ncol(prim_data)-2, "特征\n")
cat("- 辅助数据集:", nrow(aux_data), "样本\n")
cat("- 测试数据集:", nrow(test_data), "样本\n")
cat("- 真实活跃特征数:", n_active, "\n\n")

# 第二步: 训练基线Lasso模型（与auto_weibull完全一致）
cat("训练一次Lasso Cox（基线）...\n")
lasso_fit <- train_cox_lasso(train_data = prim_data, cov_names = feature_names, nfolds = 5, alpha = 1, verbose = FALSE)
lasso_coef <- as.vector(lasso_fit$coefficients)
X_test <- as.matrix(test_data[, feature_names])
lasso_risk <- as.vector(X_test %*% lasso_coef)
# 统一事件指示为0/1（TRUE=事件发生）以提升稳定性
y_test <- survival::Surv(test_data$time, test_data$status == 2)
lasso_cindex <- survival::concordance(y_test ~ lasso_risk, reverse = TRUE)$concordance
cat(sprintf("Lasso C-index: %.4f\n", lasso_cindex))

# 第三步: 参数网格设置（与auto_weibull完全一致）
cat("\n第三步: 参数网格设置\n")
lambda1_grid <- c(0.001, 0.002, 0.005, 0.01, 0.02)
lambda2_grid <- c(0.001, 0.002, 0.005, 0.01, 0.02)
lambda_beta_grid <- c(0.015, 0.02, 0.025, 0.03)
learning_rate <- 0.0012; nsteps <- 500

cat("参数搜索范围:\n")
cat("- lambda1:", paste(lambda1_grid, collapse=", "), "\n")
cat("- lambda2:", paste(lambda2_grid, collapse=", "), "\n")
cat("- lambda_beta:", paste(lambda_beta_grid, collapse=", "), "\n")
cat("- learning_rate:", learning_rate, "\n")
cat("- nsteps:", nsteps, "\n\n")

# 评估指标函数（与auto_weibull完全一致）
eval_combo <- function(transcox_coef) {
  trans_risk <- as.vector(X_test %*% transcox_coef)
  trans_cindex <- survival::concordance(y_test ~ trans_risk, reverse = TRUE)$concordance
  # 稀疏性与F1
  true_active <- abs(true_beta) > 1e-6
  trans_active <- abs(transcox_coef) > 1e-6
  tp <- sum(true_active & trans_active)
  fp <- sum(trans_active & !true_active)
  fn <- sum(true_active & !trans_active)
  precision <- ifelse((tp + fp) == 0, 0, tp / (tp + fp))
  recall <- ifelse((tp + fn) == 0, 0, tp / (tp + fn))
  f1 <- ifelse((precision + recall) == 0, 0, 2 * precision * recall / (precision + recall))
  list(cindex = trans_cindex, precision = precision, recall = recall, f1 = f1,
       nonzero = sum(trans_active))
}

# 第四步: BIC参数选择
cat("第四步: 使用BIC进行参数选择\n")
bic_start_time <- Sys.time()
bic_result <- SelParam_By_BIC_Sparse(
  primData = prim_data,
  auxData = aux_data,
  cov = feature_names,
  statusvar = "status",
  lambda1_vec = lambda1_grid,
  lambda2_vec = lambda2_grid,
  lambda_beta_vec = lambda_beta_grid,
  learning_rate = learning_rate,
  nsteps = nsteps,
  verbose = TRUE
)
bic_end_time <- Sys.time()

cat("\nBIC选择结果:\n")
cat("- 最优lambda1:", bic_result$best_lambda1, "\n")
cat("- 最优lambda2:", bic_result$best_lambda2, "\n")
cat("- 最优lambda_beta:", bic_result$best_lambda_beta, "\n")
cat("- 最小BIC值:", round(min(bic_result$BIC_array, na.rm = TRUE), 4), "\n")
cat("- BIC选择耗时:", round(difftime(bic_end_time, bic_start_time, units="mins"), 2), "分钟\n\n")

# 使用BIC选择的参数评估模型性能
cat("BIC选择的参数评估:\n")
bic_transcox_coef <- as.vector(bic_result$final_beta)
bic_metrics <- eval_combo(bic_transcox_coef)

cat("BIC模型性能:\n")
cat("- C-index:", round(bic_metrics$cindex, 4), "\n")
cat("- 选择特征数:", bic_metrics$nonzero, "\n")
cat("- F1分数:", round(bic_metrics$f1, 4), "\n")
cat("- 精确度:", round(bic_metrics$precision, 4), "\n")
cat("- 召回率:", round(bic_metrics$recall, 4), "\n")
cat("- C-index提升:", round(bic_metrics$cindex - lasso_cindex, 4), "\n\n")

# 第五步: 暴力搜索（复合评分函数，与auto_weibull完全一致）
cat("第五步: 开始网格搜索TransCox参数...\n")
brute_start_time <- Sys.time()

best <- list(score = -Inf)

for (l1 in lambda1_grid) {
  for (l2 in lambda2_grid) {
    for (lb in lambda_beta_grid) {
      cat(sprintf("- 尝试: lambda1=%.4f, lambda2=%.4f, lambda_beta=%.4f\n", l1, l2, lb))
      res <- runTransCox_Sparse(
        primData = prim_data, auxData = aux_data, cov = feature_names, statusvar = "status",
        lambda1 = l1, lambda2 = l2, lambda_beta = lb,
        learning_rate = learning_rate, nsteps = nsteps,
        auto_tune = FALSE, use_sparse = TRUE, verbose = FALSE
      )
      trans_coef <- as.vector(res$new_beta)
      m <- eval_combo(trans_coef)
      # 打分：优先提升C-index，其次F1，惩罚非零偏离真实15
      score <- (m$cindex - lasso_cindex) + 0.05 * m$f1 - 0.02 * abs(m$nonzero - n_active)
      if (score > best$score) {
        best <- list(score = score, lambda1 = l1, lambda2 = l2, lambda_beta = lb,
                     metrics = m, coef = trans_coef)
      }
    }
  }
}

brute_end_time <- Sys.time()

cat("\n=== 最佳组合 ===\n")
cat(sprintf("lambda1=%.4f, lambda2=%.4f, lambda_beta=%.4f\n", best$lambda1, best$lambda2, best$lambda_beta))
cat(sprintf("C-index: Lasso=%.4f, TransCox=%.4f (提升=%.4f)\n", lasso_cindex, best$metrics$cindex, best$metrics$cindex - lasso_cindex))
cat(sprintf("非零系数: %d (目标≈%d)\n", best$metrics$nonzero, n_active))
cat(sprintf("F1: %.3f, 精确度: %.3f, 召回率: %.3f\n", best$metrics$f1, best$metrics$precision, best$metrics$recall))
cat("- 暴力搜索耗时:", round(difftime(brute_end_time, brute_start_time, units="mins"), 2), "分钟\n\n")

# 第六步: 结果对比
cat("=== 结果对比 ===\n")
cat("方法\t\tC-index\t\t特征数\t\tF1分数\t\t精确度\t\t召回率\t\tC-index提升\n")
lasso_nonzero <- sum(abs(lasso_coef) > 1e-6)
cat("Lasso基线\t", sprintf("%.4f", lasso_cindex), "\t\t", lasso_nonzero, "\t\t", "0.0000", "\t\t", "0.0000", "\t\t", "0.0000", "\t\t", "0.0000", "\n")
cat("BIC选择\t\t", sprintf("%.4f", bic_metrics$cindex), "\t\t", bic_metrics$nonzero, "\t\t", sprintf("%.4f", bic_metrics$f1), "\t\t", sprintf("%.4f", bic_metrics$precision), "\t\t", sprintf("%.4f", bic_metrics$recall), "\t\t", sprintf("%.4f", bic_metrics$cindex - lasso_cindex), "\n")
cat("暴力搜索\t", sprintf("%.4f", best$metrics$cindex), "\t\t", best$metrics$nonzero, "\t\t", sprintf("%.4f", best$metrics$f1), "\t\t", sprintf("%.4f", best$metrics$precision), "\t\t", sprintf("%.4f", best$metrics$recall), "\t\t", sprintf("%.4f", best$metrics$cindex - lasso_cindex), "\n\n")

# 参数对比
cat("=== 参数对比 ===\n")
cat("方法\t\tlambda1\t\tlambda2\t\tlambda_beta\n")
cat("BIC选择\t\t", bic_result$best_lambda1, "\t\t", bic_result$best_lambda2, "\t\t", bic_result$best_lambda_beta, "\n")
cat("暴力搜索\t", best$lambda1, "\t\t", best$lambda2, "\t\t", best$lambda_beta, "\n\n")

# 评分对比
cat("=== 评分对比 ===\n")
bic_score <- (bic_metrics$cindex - lasso_cindex) + 0.05 * bic_metrics$f1 - 0.02 * abs(bic_metrics$nonzero - n_active)
cat("BIC选择评分:", sprintf("%.4f", bic_score), "\n")
cat("暴力搜索评分:", sprintf("%.4f", best$score), "\n")
cat("评分差异:", sprintf("%.4f", abs(bic_score - best$score)), "\n\n")

# 分析结论
cat("=== 分析结论 ===\n")
cindex_diff <- abs(bic_metrics$cindex - best$metrics$cindex)
f1_diff <- abs(bic_metrics$f1 - best$metrics$f1)
feature_diff <- abs(bic_metrics$nonzero - best$metrics$nonzero)

cat("BIC与暴力搜索的差异:\n")
cat("- C-index差异:", sprintf("%.4f", cindex_diff), "\n")
cat("- F1分数差异:", sprintf("%.4f", f1_diff), "\n")
cat("- 特征数差异:", feature_diff, "\n")
cat("- 评分差异:", sprintf("%.4f", abs(bic_score - best$score)), "\n\n")

if (cindex_diff < 0.02 && f1_diff < 0.1) {
  cat("结论: BIC选择的参数与暴力搜索结果相近，BIC方法有效！\n")
} else if (bic_metrics$cindex > lasso_cindex) {
  cat("结论: BIC选择的参数虽然不是最优，但仍能提升模型性能。\n")
} else {
  cat("结论: BIC选择的参数效果不佳，需要改进BIC计算方法。\n")
}

cat("\n实验完成！\n")