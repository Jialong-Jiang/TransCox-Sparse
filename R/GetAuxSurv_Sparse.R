#' 高维稀疏数据的源域参数估计
#' 
#' 使用Lasso-Cox回归估计源域参数，支持高维稀疏数据
#' 
#' @param auxData 源域数据
#' @param cov 协变量名称向量
#' @param lambda_aux Lasso惩罚参数，如果为NULL则自动选择
#' @param alpha 弹性网参数，1为Lasso，0为Ridge
#' @param cv_folds 交叉验证折数
#' @param weights 样本权重
#' 
#' @return 包含estR和q的列表
#' 
GetAuxSurv_Sparse <- function(auxData, cov = c("X1", "X2"), 
                             lambda_aux = NULL, alpha = 1, 
                             cv_folds = 5, weights = NULL) {
    
    cat("使用Lasso-Cox回归估计源域参数...\n")
    cat("协变量数量:", length(cov), "\n")
    cat("样本数量:", nrow(auxData), "\n")
    
    # 检查数据维度
    n_samples <- nrow(auxData)
    n_features <- length(cov)
    
    # 准备数据
    X_matrix <- as.matrix(auxData[, cov])
    time_var <- auxData$time
    status_var <- auxData$status
    
    cat("协变量矩阵维度:", nrow(X_matrix), ncol(X_matrix), "\n")
    
    # 检查数据质量
    event_rate <- mean(status_var == 2)  # 状态2表示事件
    cat("事件率:", round(event_rate, 3), "\n")
    
    if (event_rate < 0.05) {
        warning("事件率过低 (", round(event_rate, 3), ")，回退到标准Cox回归")
        return(GetAuxSurv(auxData, cov = cov))
    }
    
    # 创建生存对象
    surv_obj <- survival::Surv(time_var, status_var == 2)  # 转换为0/1编码
    
    # 设置权重
    if (is.null(weights)) {
        weights <- rep(1, n_samples)
    }
    
    # 尝试使用glmnet
    tryCatch({
        if (is.null(lambda_aux)) {
            cat("使用交叉验证选择最优lambda...\n")
            
            # 设置lambda序列
            lambda_seq <- exp(seq(log(0.001), log(1), length.out = 50))
            
            # 交叉验证
            cv_fit <- glmnet::cv.glmnet(
                x = X_matrix,
                y = surv_obj,
                family = "cox",
                alpha = alpha,
                lambda = lambda_seq,
                nfolds = min(cv_folds, n_samples),
                type.measure = "deviance",
                weights = weights
            )
            
            lambda_optimal <- cv_fit$lambda.1se
            cat("选择的lambda:", lambda_optimal, "\n")
        } else {
            lambda_optimal <- lambda_aux
            cat("使用指定的lambda:", lambda_optimal, "\n")
        }
        
        # 拟合最终模型
        lasso_fit <- glmnet::glmnet(
            x = X_matrix,
            y = surv_obj,
            family = "cox",
            alpha = alpha,
            lambda = lambda_optimal,
            weights = weights
        )
        
        # 提取系数
        estR_sparse <- as.vector(coef(lasso_fit, s = lambda_optimal))
        names(estR_sparse) <- cov
        
        # 计算基线累积风险
        # 使用标准Cox模型计算基线风险
        if (sum(abs(estR_sparse) > 1e-8) > 0) {
            # 如果有非零系数，使用这些系数
            nonzero_idx <- which(abs(estR_sparse) > 1e-8)
            if (length(nonzero_idx) > 0 && length(nonzero_idx) < n_samples - 5) {
                X_nonzero <- X_matrix[, nonzero_idx, drop = FALSE]
                cov_nonzero <- cov[nonzero_idx]
                
                cox_data <- data.frame(
                    time = time_var,
                    status = as.numeric(status_var == 2),
                    X_nonzero
                )
                colnames(cox_data)[3:ncol(cox_data)] <- cov_nonzero
                
                formula_str <- paste("survival::Surv(time, status) ~", paste(cov_nonzero, collapse = " + "))
                cox_formula <- as.formula(formula_str)
                
                cox_fit <- survival::coxph(cox_formula, data = cox_data, weights = weights)
                bhest <- survival::basehaz(cox_fit, centered = FALSE)
            } else {
                # 如果非零系数太多，使用无协变量模型
                cox_fit <- survival::coxph(survival::Surv(time_var, status_var == 2) ~ 1, weights = weights)
                bhest <- survival::basehaz(cox_fit, centered = FALSE)
            }
        } else {
            # 如果所有系数都为零，使用无协变量模型
            cox_fit <- survival::coxph(survival::Surv(time_var, status_var == 2) ~ 1, weights = weights)
            bhest <- survival::basehaz(cox_fit, centered = FALSE)
        }
        
        q <- data.frame(
            cumHazards = bhest$hazard,
            breakPoints = bhest$time
        )
        
        # 计算稀疏性统计
        nonzero_count <- sum(abs(estR_sparse) > 1e-8)
        sparsity_ratio <- 1 - nonzero_count / length(estR_sparse)
        
        cat("Lasso-Cox拟合成功:\n")
        cat("  非零系数:", nonzero_count, "/", length(estR_sparse), "\n")
        cat("  稀疏度:", round(sparsity_ratio, 3), "\n")
        
        return(list(
            estR = estR_sparse,
            q = q,
            lambda_used = lambda_optimal,
            nonzero_count = nonzero_count,
            total_features = length(estR_sparse),
            sparsity_ratio = sparsity_ratio,
            method = "lasso_cox"
        ))
        
    }, error = function(e) {
        cat("Lasso-Cox拟合失败，回退到标准Cox回归\n")
        cat("错误信息:", e$message, "\n")
        
        # 回退到标准Cox回归
        return(GetAuxSurv(auxData, cov = cov))
    })
}

#' 自动选择稀疏或标准版本的GetAuxSurv
#' 
GetAuxSurv <- function(auxData, cov = c("X1", "X2"), use_sparse = NULL) {
    
    # 自动检测是否使用稀疏版本
    if (is.null(use_sparse)) {
        n_samples <- nrow(auxData)
        n_features <- length(cov)
        event_rate <- mean(auxData$status == 2)
        
        # 如果特征数较多且事件率足够，使用稀疏版本
        use_sparse <- (n_features > 20 && event_rate > 0.1)
    }
    
    if (use_sparse) {
        return(GetAuxSurv_Sparse(auxData, cov = cov))
    }
    
    # 原始版本的GetAuxSurv
    cat("使用标准Cox回归估计源域参数...\n")
    
    # 创建公式
    formula_str <- paste("survival::Surv(time, status == 2) ~", paste(cov, collapse = " + "))
    cox_formula <- as.formula(formula_str)
    
    # 拟合Cox模型
    cox_fit <- survival::coxph(cox_formula, data = auxData)
    
    # 提取系数
    estR <- coef(cox_fit)
    if (length(estR) != length(cov)) {
        # 处理缺失系数
        full_estR <- rep(0, length(cov))
        names(full_estR) <- cov
        full_estR[names(estR)] <- estR
        estR <- full_estR
    }
    
    # 计算基线累积风险
    bhest <- survival::basehaz(cox_fit, centered = FALSE)
    q <- data.frame(
        cumHazards = bhest$hazard,
        breakPoints = bhest$time
    )
    
    return(list(
        estR = estR,
        q = q,
        method = "standard_cox"
    ))
}