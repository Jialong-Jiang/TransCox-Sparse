# 测试稀疏效果的专门脚本
# 强制使用非零lambda_beta值来验证稀疏性

library(TransCox)
library(survival)
library(glmnet)
library(reticulate)

# 配置reticulate使用正确的conda环境
use_condaenv("TransCoxEnvi", required = TRUE)
Sys.setenv(HDF5_DISABLE_VERSION_CHECK = "1")
# 加载稀疏版本的Python函数
source_python(file.path(getwd(), "inst", "python", "TransCoxFunction_Sparse.py"))

cat("=== 稀疏效果专门测试 ===\n")

# 设置随机种子
set.seed(123)

# 生成测试数据
n_samples <- 100
n_features <- 15
n_events <- round(n_samples * 0.7)

# 生成协变量矩阵
CovData <- matrix(rnorm(n_samples * n_features), nrow = n_samples, ncol = n_features)

# 生成真实的稀疏系数（只有前3个特征有效）
true_beta <- c(1.5, -1.2, 0.8, rep(0, n_features - 3))

# 生成生存时间
linear_pred <- CovData %*% true_beta
survival_times <- rexp(n_samples, rate = exp(linear_pred))
censoring_times <- rexp(n_samples, rate = 0.3)

# 观察时间和状态
obs_times <- pmin(survival_times, censoring_times)
status <- as.numeric(survival_times <= censoring_times)

cat("数据生成完成:\n")
cat("  样本数:", n_samples, "\n")
cat("  特征数:", n_features, "\n")
cat("  事件数:", sum(status), "\n")
cat("  真实非零系数数:", sum(abs(true_beta) > 1e-6), "\n")

# 使用Lasso-Cox估计源域参数
surv_obj <- Surv(obs_times, status)
lasso_fit <- cv.glmnet(CovData, surv_obj, family = "cox", alpha = 1)
estR <- as.numeric(coef(lasso_fit, s = "lambda.min"))

cat("源域Lasso-Cox结果:\n")
cat("  非零系数:", sum(abs(estR) > 1e-6), "/", length(estR), "\n")

# 准备TransCox所需的数据
fit_cox <- coxph(surv_obj ~ 1)
baseline_hazard <- basehaz(fit_cox, centered = FALSE)

# 创建累积风险和风险增量
event_times <- sort(unique(obs_times[status == 1]))
cumH <- approx(baseline_hazard$time, baseline_hazard$hazard, 
               xout = obs_times, method = "constant", 
               f = 0, yleft = 0, yright = max(baseline_hazard$hazard))$y

hazards <- diff(c(0, baseline_hazard$hazard[match(event_times, baseline_hazard$time)]))
if(length(hazards) != sum(status)) {
    hazards <- rep(mean(hazards), sum(status))
}

# 创建事件指示矩阵
Xinn <- matrix(0, nrow = n_samples, ncol = length(hazards))
event_indices <- which(status == 1)
for(i in seq_along(event_indices)) {
    Xinn[event_indices[i], i] <- 1
}

cat("\n=== 测试不同lambda_beta值的稀疏效果 ===\n")

# 测试不同的lambda_beta值
lambda_beta_values <- c(0, 0.01, 0.05, 0.1, 0.2)
lambda1 <- 0.01  # 固定较小的lambda1
lambda2 <- 0.01  # 固定较小的lambda2

results <- list()

for(i in seq_along(lambda_beta_values)) {
    lambda_beta <- lambda_beta_values[i]
    
    cat("\n--- 测试 lambda_beta =", lambda_beta, "---\n")
    
    tryCatch({
        # 检查输入数据的有效性
        cat("  检查输入数据...\n")
        cat("    CovData: ", dim(CovData), "范围:", range(CovData), "\n")
        cat("    cumH: ", length(cumH), "范围:", range(cumH), "\n")
        cat("    hazards: ", length(hazards), "范围:", range(hazards), "\n")
        cat("    estR: ", length(estR), "范围:", range(estR), "\n")
        cat("    是否有NaN:", any(is.nan(c(CovData, cumH, hazards, estR))), "\n")
        cat("    是否有Inf:", any(is.infinite(c(CovData, cumH, hazards, estR))), "\n")
        
        # 运行稀疏版本
        cat("  开始运行TransCox_Sparse...\n")
        result <- TransCox_Sparse(
            CovData = CovData,
            cumH = cumH,
            hazards = hazards,
            status = status,
            estR = estR,
            Xinn = Xinn,
            lambda1 = lambda1,
            lambda2 = lambda2,
            lambda_beta = lambda_beta,
            nsteps = 300,
            verbose = TRUE
        )
        
        # TransCox_Sparse返回4个值: eta_final, xi_final, beta_t_final, convergence_info
        eta_final <- result[[1]]
        xi_final <- result[[2]]
        beta_t_final <- result[[3]]
        conv_info <- result[[4]]
        
        # 计算稀疏性
        nonzero_beta <- sum(abs(beta_t_final) > 1e-6)
        sparsity <- 1 - nonzero_beta / length(beta_t_final)
        
        cat("结果:\n")
        cat("  非零系数:", nonzero_beta, "/", length(beta_t_final), "\n")
        cat("  稀疏度:", round(sparsity * 100, 1), "%\n")
        cat("  最大|beta_t|:", round(max(abs(beta_t_final)), 4), "\n")
        cat("  最小|beta_t|:", round(min(abs(beta_t_final)), 4), "\n")
        cat("  最终损失:", round(conv_info$final_loss, 4), "\n")
        
        results[[i]] <- list(
            lambda_beta = lambda_beta,
            nonzero_beta = nonzero_beta,
            sparsity = sparsity,
            beta_t = beta_t_final,
            convergence_info = conv_info
        )
        
    }, error = function(e) {
        cat("错误:", e$message, "\n")
        results[[i]] <- list(
            lambda_beta = lambda_beta,
            error = e$message
        )
    })
}

cat("\n=== 稀疏效果总结 ===\n")
for(i in seq_along(results)) {
    if(!is.null(results[[i]]$sparsity)) {
        cat("lambda_beta =", results[[i]]$lambda_beta, 
            ": 稀疏度 =", round(results[[i]]$sparsity * 100, 1), "%\n")
    } else {
        cat("lambda_beta =", results[[i]]$lambda_beta, ": 失败\n")
    }
}

cat("\n稀疏效果测试完成！\n")