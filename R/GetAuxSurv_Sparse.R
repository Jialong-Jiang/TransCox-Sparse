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
    
    # 使用Lasso-Cox回归估计源域参数
    
    # 检查数据维度
    n_samples <- nrow(auxData)
    n_features <- length(cov)
    
    # 准备数据
    X_matrix <- as.matrix(auxData[, cov])
    time_var <- auxData$time
    status_var <- auxData$status
    
    # 检查数据质量 - 支持0/1和1/2两种编码
    if (all(status_var %in% c(0, 1))) {
        # 0/1编码
        event_rate <- mean(status_var == 1)
        surv_obj <- survival::Surv(time_var, status_var)
    } else if (all(status_var %in% c(1, 2))) {
        # 1/2编码
        event_rate <- mean(status_var == 2)
        surv_obj <- survival::Surv(time_var, status_var == 2)
    } else {
        stop("不支持的状态编码，请使用0/1或1/2编码")
    }
    
    if (event_rate < 0.05) {
        warning("事件率过低 (", round(event_rate, 3), ")，回退到标准Cox回归")
        return(GetAuxSurv(auxData, cov = cov))
    }
    
    # 设置权重
    if (is.null(weights)) {
        weights <- rep(1, n_samples)
    }
    
    # 尝试使用glmnet
    tryCatch({
        if (is.null(lambda_aux)) {
            # 使用交叉验证选择最优lambda
            
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
        } else {
            lambda_optimal <- lambda_aux
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
        
        # Lasso-Cox拟合成功
        
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
        # Lasso-Cox拟合失败，回退到标准Cox回归
        warning("Lasso-Cox拟合失败，回退到标准Cox回归: ", e$message)
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
    
    #' 稀疏版本的辅助生存数据处理函数
    #' 
    #' @description 
    #' 处理高维稀疏辅助数据，估计源域参数
    #' 
    #' @param auxData 辅助数据集
    #' @param cov 协变量名称向量
    #' @param verbose 是否显示详细信息
    #' @return 包含估计参数的列表
    #' @export
    GetAuxSurv_Sparse <- function(auxData, cov = c("X1", "X2"), verbose = FALSE) {
        
        # 检查输入数据
        if (is.null(auxData) || nrow(auxData) == 0) {
            stop("辅助数据为空或无效")
        }
        
        if (!all(cov %in% names(auxData))) {
            missing_vars <- cov[!cov %in% names(auxData)]
            stop(paste("辅助数据中缺少协变量:", paste(missing_vars, collapse = ", ")))
        }
        
        if (!"status" %in% names(auxData)) {
            stop("辅助数据中缺少'status'列")
        }
        
        if (!"time" %in% names(auxData)) {
            stop("辅助数据中缺少'time'列")
        }
        
        # 数据预处理
        auxData <- auxData[order(auxData$time), ]
        
        # 检查事件数量
        n_events <- sum(auxData$status == 2)
        if (n_events < 5) {
            warning("辅助数据中事件数量过少，可能影响估计精度")
        }
        
        # 检查数据维度
        n_samples <- nrow(auxData)
        n_features <- length(cov)
        
        if (verbose) {
            cat("辅助数据维度: 样本数 =", n_samples, ", 特征数 =", n_features, "\n")
            cat("事件数量:", n_events, "\n")
        }
        
        # 高维稀疏数据的特殊处理
        if (n_features > n_samples / 2) {
            if (verbose) {
                cat("检测到高维数据，使用稀疏估计方法\n")
            }
            
            # 使用正则化Cox回归
            if (!requireNamespace("glmnet", quietly = TRUE)) {
                stop("高维数据需要安装'glmnet'包")
            }
            
            # 准备数据
            x_matrix <- as.matrix(auxData[, cov])
            y_surv <- survival::Surv(auxData$time, auxData$status == 2)
            
            # 使用交叉验证选择lambda
            cv_fit <- glmnet::cv.glmnet(x_matrix, y_surv, family = "cox", alpha = 1)
            
            # 使用最优lambda拟合模型
            fit <- glmnet::glmnet(x_matrix, y_surv, family = "cox", alpha = 1, lambda = cv_fit$lambda.min)
            
            # 提取系数
            estR <- as.numeric(coef(fit))
            
            # 计算基线风险
            linear_pred <- as.numeric(x_matrix %*% estR)
            
            # 使用简化的基线风险估计
            event_times <- sort(unique(auxData$time[auxData$status == 2]))
            q <- rep(0, length(event_times))
            
            for (i in seq_along(event_times)) {
                t <- event_times[i]
                at_risk <- auxData$time >= t
                events_at_t <- auxData$time == t & auxData$status == 2
                
                if (sum(at_risk) > 0 && sum(events_at_t) > 0) {
                    risk_sum <- sum(exp(linear_pred[at_risk]))
                    q[i] <- sum(events_at_t) / risk_sum
                }
            }
            
            # 创建时间-风险映射
            time_risk_map <- data.frame(time = event_times, q = q)
            
        } else {
            # 标准维度数据的处理
            if (verbose) {
                cat("使用标准Cox回归估计\n")
            }
            
            # 使用标准Cox回归
            formula_str <- paste("survival::Surv(time, status == 2) ~", paste(cov, collapse = " + "))
            cox_formula <- as.formula(formula_str)
            
            fit <- survival::coxph(cox_formula, data = auxData)
            
            # 提取系数
            estR <- coef(fit)
            if (any(is.na(estR))) {
                warning("部分系数估计为NA，可能存在共线性问题")
                estR[is.na(estR)] <- 0
            }
            
            # 计算基线风险
            basehaz <- survival::basehaz(fit)
            
            # 创建时间-风险映射
            time_risk_map <- data.frame(
                time = basehaz$time,
                q = diff(c(0, basehaz$hazard))
            )
        }
        
        # 数据质量检查
        if (any(is.na(estR))) {
            warning("系数估计中包含NA值")
            estR[is.na(estR)] <- 0
        }
        
        if (any(is.infinite(estR))) {
            warning("系数估计中包含无穷值")
            estR[is.infinite(estR)] <- 0
        }
        
        # 稀疏性统计
        nonzero_count <- sum(abs(estR) > 1e-8)
        sparsity_ratio <- 1 - nonzero_count / length(estR)
        
        if (verbose) {
            cat("非零系数数量:", nonzero_count, "/", length(estR), "\n")
            cat("稀疏度:", round(sparsity_ratio * 100, 2), "%\n")
        }
        
        # 返回结果
        result <- list(
            estR = estR,
            q = time_risk_map,
            n_events = n_events,
            n_samples = n_samples,
            n_features = n_features,
            nonzero_count = nonzero_count,
            sparsity_ratio = sparsity_ratio,
            method = if (n_features > n_samples / 2) "sparse_regularized" else "standard_cox"
        )
        
        return(result)
    }
    
    return(list(
        estR = estR,
        q = q,
        method = "standard_cox"
    ))
}