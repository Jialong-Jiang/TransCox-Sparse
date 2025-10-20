#' 高维稀疏TransCox主函数
#' 
#' 整合所有组件的主要接口函数，支持高维稀疏数据
#' 
#' @param primData 目标域数据
#' @param auxData 源域数据
#' @param cov 协变量名称向量
#' @param statusvar 状态变量名称
#' @param lambda1 eta的L1惩罚参数（如果为NULL则自动选择）
#' @param lambda2 xi的L1惩罚参数（如果为NULL则自动选择）
#' @param lambda_beta beta_t的L1惩罚参数（如果为NULL则自动选择）
#' @param learning_rate 学习率
#' @param nsteps 优化步数
#' @param auto_tune 是否自动调参
#' @param use_sparse 是否强制使用稀疏版本
#' @param verbose 是否显示详细信息
#' 
#' @return TransCox结果列表
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
                              verbose = TRUE) {
    
    if (verbose) {
        cat("=== TransCox高维稀疏分析 ===\n")
        cat("目标域样本数:", nrow(primData), "\n")
        cat("源域样本数:", nrow(auxData), "\n")
        cat("特征数:", length(cov), "\n")
    }
    
    # 自动检测是否使用稀疏版本
    if (is.null(use_sparse)) {
        n_samples <- nrow(primData)
        n_features <- length(cov)
        use_sparse <- (n_features > n_samples / 2)
        
        if (verbose) {
            if (use_sparse) {
                cat("检测到高维数据 (p=", n_features, ", n=", n_samples, 
                    ")，使用稀疏版本\n")
            } else {
                cat("使用标准版本\n")
            }
        }
    }
    
    # 加载必要的函数
    if (use_sparse) {
        source(file.path(getwd(), "R", "GetAuxSurv_Sparse.R"))
        source(file.path(getwd(), "R", "SelParam_By_BIC_Sparse.R"))
        source_python(file.path(getwd(), "inst", "python", "TransCoxFunction_Sparse.py"))
    } else {
        source_python(system.file("python", "TransCoxFunction.py", package = "TransCox"))
    }
    
    # 参数调优
    if (auto_tune && (is.null(lambda1) || is.null(lambda2) || is.null(lambda_beta))) {
        if (verbose) cat("\n=== 自动参数调优 ===\n")
        
        if (use_sparse) {
            # 使用稀疏版本的BIC选择
            bic_result <- SelParam_By_BIC_Sparse(
                primData = primData,
                auxData = auxData,
                cov = cov,
                statusvar = statusvar,
                lambda1_vec = if(is.null(lambda1)) c(0.01, 0.05, 0.1, 0.2, 0.5, 1.0) else lambda1,
                lambda2_vec = if(is.null(lambda2)) c(0.01, 0.05, 0.1, 0.2, 0.5, 1.0) else lambda2,
                lambda_beta_vec = if(is.null(lambda_beta)) c(0, 0.01, 0.05, 0.1, 0.2, 0.5) else lambda_beta,
                learning_rate = learning_rate,
                nsteps = nsteps,
                verbose = verbose
            )
            
            lambda1 <- bic_result$best_lambda1
            lambda2 <- bic_result$best_lambda2
            lambda_beta <- bic_result$best_lambda_beta
            
            if (verbose) {
                cat("BIC选择的最优参数:\n")
                cat("  lambda1:", lambda1, "\n")
                cat("  lambda2:", lambda2, "\n") 
                cat("  lambda_beta:", lambda_beta, "\n")
            }
            
            # 直接返回BIC结果中的最终结果
            result <- list(
                eta = bic_result$final_eta,
                xi = bic_result$final_xi,
                new_beta = bic_result$final_beta,
                new_IntH = bic_result$final_xi + auxData$time[auxData$status == 2],  # 需要调整
                source_estR = bic_result$source_estR,
                lambda1_used = lambda1,
                lambda2_used = lambda2,
                lambda_beta_used = lambda_beta,
                convergence_info = bic_result$convergence_info,
                bic_result = bic_result
            )
            
            return(result)
            
        } else {
            # 使用原始版本的BIC选择
            bic_result <- SelParam_By_BIC(
                primData = primData,
                auxData = auxData,
                cov = cov,
                statusvar = statusvar,
                lambda1_vec = if(is.null(lambda1)) c(0.1, 0.5, seq(1, 5, by = 0.5)) else lambda1,
                lambda2_vec = if(is.null(lambda2)) c(0.1, 0.5, seq(1, 5, by = 0.5)) else lambda2,
                learning_rate = learning_rate,
                nsteps = nsteps
            )
            
            lambda1 <- bic_result$best_la1
            lambda2 <- bic_result$best_la2
            lambda_beta <- 0  # 原始版本不使用lambda_beta
        }
    }
    
    # 设置默认值
    if (is.null(lambda1)) lambda1 <- 0.1
    if (is.null(lambda2)) lambda2 <- 0.1
    if (is.null(lambda_beta)) lambda_beta <- 0
    
    if (verbose) {
        cat("\n=== 最终模型拟合 ===\n")
        cat("使用参数: lambda1=", lambda1, ", lambda2=", lambda2, ", lambda_beta=", lambda_beta, "\n")
    }
    
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
    
    # 运行TransCox
    if (use_sparse && lambda_beta > 0) {
        # 使用稀疏版本
        trans_result <- TransCox_Sparse(
            CovData = as.matrix(CovData),
            cumH = cumH,
            hazards = hazards,
            status = status,
            estR = Pout$estR,
            Xinn = Pout$Xinn,
            lambda1 = lambda1,
            lambda2 = lambda2,
            lambda_beta = lambda_beta,
            learning_rate = learning_rate,
            nsteps = nsteps,
            verbose = verbose
        )
        
        eta <- trans_result[[1]]
        xi <- trans_result[[2]]
        new_beta <- trans_result[[3]]
        convergence_info <- trans_result[[4]]
        
    } else {
        # 使用原始版本
        trans_result <- TransCox(
            CovData = as.matrix(CovData),
            cumH = cumH,
            hazards = hazards,
            status = status,
            estR = Pout$estR,
            Xinn = Pout$Xinn,
            lambda1 = lambda1,
            lambda2 = lambda2,
            learning_rate = learning_rate,
            nsteps = nsteps
        )
        
        eta <- trans_result[[1]]
        xi <- trans_result[[2]]
        new_beta <- Pout$estR + eta
        convergence_info <- NULL
    }
    
    # 计算稀疏性统计
    nonzero_beta <- sum(abs(new_beta) > 1e-8)
    sparsity_ratio <- 1 - nonzero_beta / length(new_beta)
    
    if (verbose) {
        cat("\n=== 结果摘要 ===\n")
        cat("最终beta非零系数:", nonzero_beta, "/", length(new_beta), "\n")
        cat("稀疏度:", round(sparsity_ratio * 100, 2), "%\n")
        
        if (!is.null(convergence_info)) {
            cat("优化收敛:", ifelse(convergence_info$converged, "是", "否"), "\n")
            cat("最终损失:", round(convergence_info$final_loss, 6), "\n")
        }
    }
    
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
        use_sparse = use_sparse
    )
    
    class(result) <- "TransCox_Sparse"
    return(result)
}

#' 打印TransCox_Sparse结果
#' 
print.TransCox_Sparse <- function(x, ...) {
    cat("TransCox高维稀疏分析结果\n")
    cat("========================\n")
    cat("特征数:", length(x$new_beta), "\n")
    cat("非零系数:", x$nonzero_count, "\n")
    cat("稀疏度:", round(x$sparsity_ratio * 100, 2), "%\n")
    cat("使用稀疏版本:", x$use_sparse, "\n")
    cat("\n参数:\n")
    cat("  lambda1 (eta):", x$lambda1_used, "\n")
    cat("  lambda2 (xi):", x$lambda2_used, "\n")
    cat("  lambda_beta:", x$lambda_beta_used, "\n")
    
    if (!is.null(x$convergence_info)) {
        cat("\n收敛信息:\n")
        cat("  收敛:", ifelse(x$convergence_info$converged, "是", "否"), "\n")
        cat("  最终损失:", round(x$convergence_info$final_loss, 6), "\n")
        cat("  优化步数:", x$convergence_info$steps_taken, "\n")
    }
    
    cat("\n前10个系数:\n")
    print(head(x$new_beta, 10))
}

#' 向后兼容的runTransCox_one函数
#' 
runTransCox_one <- function(Pout, l1 = 1, l2 = 1, learning_rate = 0.004, nsteps = 200,
                           cov = c('X1', 'X2'), lambda_beta = 0, use_sparse = NULL) {
    
    # 自动检测是否使用稀疏版本
    if (is.null(use_sparse)) {
        n_features <- length(cov)
        use_sparse <- (n_features > 50 || lambda_beta > 0)  # 如果特征数>50或使用lambda_beta则使用稀疏版本
    }
    
    if (use_sparse && lambda_beta > 0) {
        # 加载稀疏版本
        source_python(file.path(getwd(), "inst", "python", "TransCoxFunction_Sparse.py"))
        
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
        TransCox <- NULL
        .onLoad <- function(libname, pkgname) {
            tf <<- reticulate::import("tensorflow", delay_load = TRUE)
            tfp <<- reticulate::import("tensorflow_probability", delay_load = TRUE)
            np <<- reticulate::import("numpy", delay_load = TRUE)
            source_python(system.file("python", "TransCoxFunction.py", package = "TransCox"))
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