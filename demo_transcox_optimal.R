
# =============================================================================

# 清理环境
rm(list = ls())
gc()
library(reticulate)
use_python("D:/anaconda3/envs/TransCoxEnvi/python.exe",required = TRUE)
use_condaenv("TransCoxEnvi")
library(TransCox)
source_python(system.file("python", "TransCoxFunction.py", package = "TransCox"))
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

cat("Transcox模型性能:\n")
cat("- C-index:", round(bic_metrics$cindex, 4), "\n")
cat("- 选择特征数:", bic_metrics$nonzero, "\n")
cat("- F1分数:", round(bic_metrics$f1, 4), "\n")
cat("- 精确度:", round(bic_metrics$precision, 4), "\n")
cat("- 召回率:", round(bic_metrics$recall, 4), "\n")
cat("- C-index提升:", round(bic_metrics$cindex - lasso_cindex, 4), "\n\n")


cat("--- 系数向量与真实Beta的比较 ---\n\n")

# 确保所有向量长度一致 (p)
cat("--- 系数向量与真实Beta的详细比较 ---\n\n")


# 创建一个对比数据框
active_comparison <- data.frame(
  Index = active_indices,
  TrueBeta = true_beta[active_indices],
  LassoEst = lasso_coef[active_indices],
  TransCoxEst = bic_transcox_coef[active_indices]
)
# 计算估计误差
active_comparison$LassoError <- active_comparison$LassoEst - active_comparison$TrueBeta
active_comparison$TransCoxError <- active_comparison$TransCoxEst - active_comparison$TrueBeta

cat("活跃系数的估计值及误差:\n")
print(round(active_comparison, 5)) # 打印表格，保留5位小数
cat("\n")

# 1. 计算Lasso的误差
lasso_mse <- mean((lasso_coef - true_beta)^2)
lasso_mae <- mean(abs(lasso_coef - true_beta))

# 2. 计算TransCox-BIC的误差
bic_mse <- mean((bic_transcox_coef - true_beta)^2)
bic_mae <- mean(abs(bic_transcox_coef - true_beta))

# 3. 打印比较结果
cat(sprintf("Lasso Cox (基线) vs True Beta:\n"))
cat(sprintf("- 均方误差 (MSE): %.6f\n", lasso_mse))
cat(sprintf("- 平均绝对误差 (MAE): %.6f\n", lasso_mae))

cat(sprintf("\nTransCox-BIC vs True Beta:\n"))
cat(sprintf("- 均方误差 (MSE): %.6f\n", bic_mse))
cat(sprintf("- 平均绝对误差 (MAE): %.6f\n", bic_mae))
