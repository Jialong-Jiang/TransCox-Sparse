#' Sparse TransCox Model for High-Dimensional Survival Analysis
#' 
#' @description 
#' This function implements the sparse TransCox model for high-dimensional survival analysis
#' with transfer learning. It integrates source domain (auxiliary) data to improve prediction
#' performance on the target domain (primary) data through regularized Cox regression with
#' L1 penalties.
#' 
#' @details
#' The sparse TransCox model addresses the challenge of limited sample sizes in survival
#' analysis by leveraging information from related source domains. The model uses L1
#' regularization to achieve sparsity and automatic feature selection, making it suitable
#' for high-dimensional data where the number of features exceeds the number of samples.
#' 
#' @param primData A data.frame containing the target domain survival data. Must include
#'   survival time, event status, and covariates.
#' @param auxData A data.frame containing the source domain survival data with the same
#'   structure as primData.
#' @param cov A character vector specifying the names of covariates to be used in the model.
#'   Default is c("X1", "X2").
#' @param statusvar A character string specifying the name of the event status variable.
#'   Default is "status".
#' @param lambda1 Numeric. L1 penalty parameter for eta (auxiliary parameter). If NULL,
#'   will be automatically selected via BIC. Default is NULL.
#' @param lambda2 Numeric. L1 penalty parameter for xi (transfer parameter). If NULL,
#'   will be automatically selected via BIC. Default is NULL.
#' @param lambda_beta Numeric. L1 penalty parameter for beta_t (target parameter). If NULL,
#'   will be automatically selected via BIC. Default is NULL.
#' @param learning_rate Numeric. Learning rate for the optimization algorithm. Default is 0.004.
#' @param nsteps Integer. Maximum number of optimization steps. Default is 200.
#' @param auto_tune Logical. Whether to automatically tune hyperparameters using BIC.
#'   Default is TRUE.
#' @param use_sparse Logical. Whether to force the use of sparse implementation. If NULL,
#'   automatically determined based on data dimensions. Default is NULL.
#' @param verbose Logical. Whether to display detailed progress information. Default is TRUE.
#' @param tolerance Numeric. Convergence tolerance for the optimization algorithm.
#'   Default is 1e-6.
#' @param early_stopping Logical. Whether to enable early stopping mechanism. Default is TRUE.
#' @param adaptive_lr Logical. Whether to use adaptive learning rate. Default is TRUE.
#' 
#' @return A list containing the following components:
#' \describe{
#'   \item{beta_t}{Estimated coefficients for the target domain}
#'   \item{eta}{Estimated auxiliary parameters}
#'   \item{xi}{Estimated transfer parameters}
#'   \item{lambda1}{Used L1 penalty for eta}
#'   \item{lambda2}{Used L1 penalty for xi}
#'   \item{lambda_beta}{Used L1 penalty for beta_t}
#'   \item{convergence}{Convergence information}
#'   \item{sparse_info}{Information about sparsity patterns}
#' }
#' 
#' @examples
#' \dontrun{
#' # Generate example data
#' data <- generate_sparse_survival_data(n_main = 100, n_aux = 200, p = 50)
#' 
#' # Fit sparse TransCox model
#' result <- runTransCox_Sparse(
#'   primData = data$prim_data,
#'   auxData = data$aux_data,
#'   cov = paste0("X", 1:50),
#'   statusvar = "status"
#' )
#' 
#' # View results
#' print(result)
#' }
#' 
#' @export
#' 
runTransCox_Sparse <- function(primData, auxData, 
                              cov = c("X1", "X2"),
                              statusvar = "status",
                              lambda1 = NULL,
                              lambda2 = NULL, 
                              lambda_beta = NULL,
                              learning_rate = 0.004,
                              nsteps = 200,
                              auto_tune = TRUE,
                              use_sparse = NULL,
                              verbose = TRUE,
                              tolerance = 1e-6,
                              early_stopping = TRUE,
                              adaptive_lr = TRUE,
                              parallel = FALSE,  # 默认关闭并行计算
                              n_cores = NULL) {
    
    # 参数说明：
    # parallel: 是否使用并行计算进行参数搜索，默认FALSE
    # 警告：并行计算在某些环境下可能出现问题（如服务器、RStudio）
    # 只有在确定环境支持时才启用
    # n_cores: 并行计算使用的核心数，如果为NULL则使用detectCores()-1
    
    # 基本信息记录（可选）
    
    # 自动检测是否使用稀疏版本
    if (is.null(use_sparse)) {
        n_samples <- nrow(primData)
        n_features <- length(cov)
        use_sparse <- (n_features > n_samples / 2)
        
        # 自动选择稀疏版本或标准版本
    }
    
    # 加载必要的函数
    if (!exists("GetPrimaryParam")) {
        source(file.path(getwd(), "R", "GetPrimaryParam.R"))
    }
    if (!exists("deltaQ")) {
        source(file.path(getwd(), "R", "deltaQ.R"))
    }
    
    if (use_sparse) {
        if (!exists("GetAuxSurv_Sparse")) {
            source(file.path(getwd(), "R", "GetAuxSurv_Sparse.R"))
        }
        if (!exists("SelParam_By_BIC_Sparse")) {
            source(file.path(getwd(), "R", "SelParam_By_BIC_Sparse.R"))
        }
        if (!exists("TransCox_Sparse")) {
            reticulate::source_python(file.path(getwd(), "inst", "python", "TransCoxFunction_Sparse.py"))
        }
    } else {
        if (!exists("TransCox")) {
            reticulate::source_python(system.file("python", "TransCoxFunction.py", package = "TransCox"))
        }
    }
    
    # 参数调优
    # 如果auto_tune=TRUE且任何参数是向量或NULL，则进行BIC选择
    need_tune <- auto_tune && (
        is.null(lambda1) || is.null(lambda2) || is.null(lambda_beta) ||
        length(lambda1) > 1 || length(lambda2) > 1 || length(lambda_beta) > 1
    )
    
    if (need_tune) {
        # 自动参数调优
        
        if (use_sparse) {
            # 使用稀疏版本的BIC选择
            bic_result <- SelParam_By_BIC_Sparse(
                primData = primData,
                auxData = auxData,
                cov = cov,
                statusvar = statusvar,
                lambda1_vec = if(is.null(lambda1)) c(0.1, 0.5, 1.0, 2.0) else lambda1,  # 分层搜索：粗搜索
        lambda2_vec = if(is.null(lambda2)) c(0.1, 0.5, 1.0, 2.0) else lambda2,  # 分层搜索：粗搜索
        lambda_beta_vec = if(is.null(lambda_beta)) c(0, 1e-3, 1e-2, 5e-2) else lambda_beta,  # 分层搜索：粗搜索
                learning_rate = learning_rate,
                nsteps = nsteps,
                verbose = verbose,
                parallel = parallel,
                n_cores = n_cores
            )
            
            lambda1 <- bic_result$best_lambda1
            lambda2 <- bic_result$best_lambda2
            lambda_beta <- bic_result$best_lambda_beta
            
            # BIC选择完成
            
            # 如果BIC结果包含最终模型，直接返回
            if (!is.null(bic_result$final_beta)) {
                result <- list(
                    eta = bic_result$final_eta,
                    xi = bic_result$final_xi,
                    new_beta = bic_result$final_beta,
                    new_IntH = if (!is.null(bic_result$final_xi)) bic_result$final_xi else rep(0, sum(auxData$status == 2)),
                    source_estR = if (!is.null(bic_result$source_estR)) bic_result$source_estR else rep(0, length(cov)),
                    lambda1_used = lambda1,
                    lambda2_used = lambda2,
                    lambda_beta_used = lambda_beta,
                    convergence_info = bic_result$convergence_info,
                    bic_result = bic_result,
                    nonzero_count = sum(abs(bic_result$final_beta) > 1e-8),
                    sparsity_ratio = 1 - sum(abs(bic_result$final_beta) > 1e-8) / length(bic_result$final_beta),
                    use_sparse = use_sparse
                )
                
                class(result) <- "TransCox_Sparse"
                return(result)
            }
            
        } else {
            # 使用原始版本的BIC选择
            bic_result <- SelParam_By_BIC(
                primData = primData,
                auxData = auxData,
                cov = cov,
                statusvar = statusvar,
                lambda1_vec = if(is.null(lambda1)) c(0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1) else lambda1,
                lambda2_vec = if(is.null(lambda2)) c(0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1) else lambda2,
                learning_rate = learning_rate,
                nsteps = nsteps
            )
            
            lambda1 <- bic_result$best_la1
            lambda2 <- bic_result$best_la2
            lambda_beta <- 0  # 原始版本不使用lambda_beta
        }
    }
    
    # 设置默认值
    if (is.null(lambda1)) lambda1 <- 0.01  # 降低默认值，减少过度正则化
    if (is.null(lambda2)) lambda2 <- 0.01  # 降低默认值，减少过度正则化
    if (is.null(lambda_beta)) {
        # 根据数据维度自适应设置lambda_beta默认值
        n_features <- length(cov)
        n_samples <- nrow(primData)
        if (use_sparse && n_features > n_samples / 2) {
            lambda_beta <- 0.03  # 高维数据使用适度的稀疏化
        } else if (use_sparse) {
            lambda_beta <- 0.02  # 稀疏模式下使用轻度稀疏化
        } else {
            lambda_beta <- 0
        }
    }
    
    # 自适应学习率调整
    if (adaptive_lr && nsteps > 100) {
        learning_rate <- learning_rate * 0.8
        if (verbose) cat("自适应学习率调整为:", learning_rate, "\n")
    }
    
    # 早停机制
    if (early_stopping && nsteps > 500) {
        nsteps <- min(nsteps, 300)
        if (verbose) cat("早停机制启用，最大步数调整为:", nsteps, "\n")
    }
    
    # 根据数据复杂度调整最大步数
    n_features <- length(cov)
    if (n_features > 500) {
        nsteps <- min(nsteps * 1.5, 500)  # 高维数据可能需要更多步数
    } else if (n_features < 50) {
        nsteps <- max(nsteps * 0.7, 100)  # 低维数据通常收敛更快
    }
    
    if (verbose) {
        cat("早停机制启用，最大步数调整为:", nsteps, "\n")
    }
    
    # 最终模型拟合
    
    # 估计源域参数
    if (use_sparse) {
        Cout <- GetAuxSurv_Sparse(auxData, cov = cov)
    } else {
        Cout <- GetAuxSurv(auxData, cov = cov)
    }
    
    # 计算目标域参数
    Pout <- GetPrimaryParam(primData, q = Cout$q, estR = Cout$estR)
    
    # 准备数据
    CovData <- Pout$primData[, cov]
    status <- Pout$primData[, statusvar]
    cumH <- Pout$primData$fullCumQ
    hazards <- Pout$dQ$dQ
    
    # 优化的R-Python接口调用
    # 数据预处理和类型优化
    if (verbose) cat("📊 预处理数据以优化传输...\n")
    
    # 预处理数据矩阵，确保连续内存布局和正确类型
    CovData_optimized <- as.matrix(CovData, mode = "double")
    storage.mode(CovData_optimized) <- "double"
    
    # 预处理向量数据
    cumH_optimized <- as.double(cumH)
    hazards_optimized <- as.double(hazards)
    status_optimized <- as.integer(status)
    estR_optimized <- as.double(Pout$estR)
    Xinn_optimized <- as.matrix(Pout$Xinn, mode = "double")
    storage.mode(Xinn_optimized) <- "double"
    
    # 检查是否使用稀疏版本（处理向量参数的情况）
    use_sparse_version <- use_sparse && (
        (length(lambda_beta) > 1) || 
        (length(lambda_beta) == 1 && lambda_beta > 0)
    )
    
    if (use_sparse_version) {
        # 创建优化的参数包，减少函数调用开销
        params_package <- list(
            CovData = CovData_optimized,
            cumH = cumH_optimized,
            hazards = hazards_optimized,
            status = status_optimized,
            estR = estR_optimized,
            Xinn = Xinn_optimized,
            lambda1 = as.double(lambda1),
            lambda2 = as.double(lambda2),
            lambda_beta = as.double(lambda_beta),
            learning_rate = as.double(learning_rate),
            nsteps = as.integer(nsteps),
            tolerance = as.double(tolerance),
            verbose = verbose
        )
        
        # 使用稀疏版本 - 单次批量传输
        if (verbose) cat("🚀 调用优化的稀疏TransCox函数...\n")
        trans_result <- do.call(TransCox_Sparse, params_package)
        
        eta <- trans_result[[1]]
        xi <- trans_result[[2]]
        new_beta <- trans_result[[3]]
        convergence_info <- trans_result[[4]]
        
    } else {
        # 创建优化的参数包（原始版本）
        params_package_orig <- list(
            CovData = CovData_optimized,
            cumH = cumH_optimized,
            hazards = hazards_optimized,
            status = status_optimized,
            estR = estR_optimized,
            Xinn = Xinn_optimized,
            lambda1 = as.double(lambda1),
            lambda2 = as.double(lambda2),
            learning_rate = as.double(learning_rate),
            nsteps = as.integer(nsteps)
        )
        
        # 使用原始版本 - 单次批量传输
        if (verbose) cat("🚀 调用优化的原始TransCox函数...\n")
        trans_result <- do.call(TransCox, params_package_orig)
        
        eta <- trans_result[[1]]
        xi <- trans_result[[2]]
        new_beta <- estR_optimized + eta  # 使用预处理的数据
        convergence_info <- NULL
    }
    
    # 计算稀疏性统计
    nonzero_beta <- sum(abs(new_beta) > 1e-8, na.rm = TRUE)
    sparsity_ratio <- 1 - nonzero_beta / length(new_beta)
    
    # 防止过度稀疏化的安全检查
    sparsity_warnings <- character(0)
    if (!is.na(nonzero_beta) && nonzero_beta == 0) {
        sparsity_warnings <- c(sparsity_warnings, "警告: 所有系数被置零，模型可能过度稀疏化")
        if (verbose) cat("⚠️  警告: 所有系数被置零，建议降低lambda_beta值\n")
    } else if (!is.na(sparsity_ratio) && sparsity_ratio > 0.95) {
         sparsity_warnings <- c(sparsity_warnings, paste0("警告: 稀疏度过高 (", round(sparsity_ratio * 100, 1), "%)"))
         if (verbose) cat("⚠️  警告: 稀疏度过高，可能影响模型性能\n")
     } else if (!is.na(sparsity_ratio) && sparsity_ratio < 0.1 && use_sparse) {
         sparsity_warnings <- c(sparsity_warnings, "提示: 稀疏度较低，可能需要增加lambda_beta")
         if (verbose) cat("💡 提示: 稀疏度较低，考虑增加lambda_beta以获得更好的特征选择\n")
     }
     
     # 最小非零系数检查
     min_nonzero <- max(1, round(length(new_beta) * 0.05))  # 至少保留5%的特征
     if (!is.na(nonzero_beta) && nonzero_beta > 0 && nonzero_beta < min_nonzero && use_sparse) {
        sparsity_warnings <- c(sparsity_warnings, paste0("建议: 非零系数过少 (", nonzero_beta, "), 建议至少保留 ", min_nonzero, " 个"))
        if (verbose) cat("💡 建议: 考虑降低lambda_beta以保留更多有用特征\n")
    }
    
    # 计算完成
    
    # 返回结果
    result <- list(
        eta = eta,
        xi = xi,
        new_beta = new_beta,
        new_IntH = Pout$dQ$dQ + xi,
        time = Pout$primData[status == 2, "time"],
        source_estR = Pout$estR,
        lambda1_used = lambda1,
        lambda2_used = lambda2,
        lambda_beta_used = lambda_beta,
        nonzero_count = nonzero_beta,
        sparsity_ratio = sparsity_ratio,
        convergence_info = convergence_info,
        sparsity_warnings = sparsity_warnings,
        use_sparse = use_sparse
    )
    
    class(result) <- "TransCox_Sparse"
    return(result)
}

#' 打印TransCox_Sparse结果
#' 
print.TransCox_Sparse <- function(x, ...) {
    cat("TransCox Sparse Results\n")
    cat("Features:", length(x$new_beta), "\n")
    cat("Non-zero coefficients:", x$nonzero_count, "\n")
    cat("Sparsity:", round(x$sparsity_ratio * 100, 2), "%\n")
    cat("Parameters: lambda1=", x$lambda1_used, ", lambda2=", x$lambda2_used, ", lambda_beta=", x$lambda_beta_used, "\n")
    
    if (!is.null(x$convergence_info)) {
        cat("Converged:", ifelse(x$convergence_info$converged, "Yes", "No"), "\n")
    }
}

#' 向后兼容的runTransCox_one函数
#' 
runTransCox_one <- function(Pout, l1 = 1, l2 = 1, learning_rate = 0.004, nsteps = 200,
                           cov = c('X1', 'X2'), lambda_beta = 0, use_sparse = NULL) {
    
    # 自动检测是否使用稀疏版本
    if (is.null(use_sparse)) {
        n_features <- length(cov)
        has_lambda_beta <- (length(lambda_beta) > 1) || (length(lambda_beta) == 1 && lambda_beta > 0)
        use_sparse <- (n_features > 50 || has_lambda_beta)  # 如果特征数>50或使用lambda_beta则使用稀疏版本
    }
    
    # 检查是否使用稀疏版本（处理向量参数的情况）
    use_sparse_version <- use_sparse && (
        (length(lambda_beta) > 1) || 
        (length(lambda_beta) == 1 && lambda_beta > 0)
    )
    
    if (use_sparse_version) {
        # 加载稀疏版本
        if (!exists("TransCox_Sparse")) {
            reticulate::source_python(file.path(getwd(), "inst", "python", "TransCoxFunction_Sparse.py"))
        }
        
        CovData <- Pout$primData[, cov]
        status <- Pout$primData[, "status"]
        cumH <- Pout$primData$fullCumQ
        hazards <- Pout$dQ$dQ
        
        test <- TransCox_Sparse(CovData = as.matrix(CovData),
                               cumH = cumH,
                               hazards = hazards,
                               status = status,
                               estR = Pout$estR,
                               Xinn = Pout$Xinn,
                               lambda1 = l1, lambda2 = l2, lambda_beta = lambda_beta,
                               learning_rate = learning_rate,
                               nsteps = nsteps,
                               verbose = FALSE)
        
        return(list(eta = test[[1]],
                   xi = test[[2]],
                   new_beta = test[[3]],  # 直接使用稀疏版本的beta_t
                   new_IntH = Pout$dQ$dQ + test[[2]],
                   time = Pout$primData[status == 2, "time"]))
        
    } else {
        # 使用原始版本
        if (!exists("TransCox")) {
            tryCatch({
                reticulate::source_python(system.file("python", "TransCoxFunction.py", package = "TransCox"))
            }, error = function(e) {
                # 如果包路径不存在，尝试本地路径
                reticulate::source_python(file.path(getwd(), "inst", "python", "TransCoxFunction.py"))
            })
        }
        
        CovData = Pout$primData[, cov]
        status = Pout$primData[, "status"]
        cumH = Pout$primData$fullCumQ
        hazards = Pout$dQ$dQ
        
        test <- TransCox(CovData = as.matrix(CovData),
                        cumH = cumH,
                        hazards = hazards,
                        status = status,
                        estR = Pout$estR,
                        Xinn = Pout$Xinn,
                        lambda1 = l1, lambda2 = l2,
                        learning_rate = learning_rate,
                        nsteps = nsteps)
        names(test) <- c("eta", "xi")
        
        return(list(eta = test$eta,
                   xi = test$xi,
                   new_beta = Pout$estR + test$eta,
                   new_IntH = Pout$dQ$dQ + test$xi,
                   time = Pout$primData[status == 2, "time"]))
    }
}