#' 高维稀疏版本的BIC参数选择函数
#' 
#' 扩展原始BIC选择以支持lambda_beta参数调优
#' 
#' @param primData 目标域数据
#' @param auxData 源域数据  
#' @param cov 协变量名称向量
#' @param statusvar 状态变量名称
#' @param lambda1_vec eta的L1惩罚参数候选值
#' @param lambda2_vec xi的L1惩罚参数候选值
#' @param lambda_beta_vec beta_t的L1惩罚参数候选值（新增）
#' @param learning_rate 学习率
#' @param nsteps 优化步数
#' @param use_sparse 是否使用稀疏版本
#' @param parallel 是否使用并行计算
#' @param verbose 是否显示详细信息
#' 
#' @return 包含最优参数和BIC矩阵的列表
#' 
SelParam_By_BIC_Sparse <- function(primData, auxData, cov = c("X1", "X2"),
                                  statusvar = "status",
                                  lambda1_vec = c(0.1, 0.5, seq(1, 5, by = 0.5)),
                                  lambda2_vec = c(0.1, 0.5, seq(1, 5, by = 0.5)),
                                  lambda_beta_vec = c(0, 0.01, 0.05, 0.1, 0.2, 0.5),
                                  learning_rate = 0.004,
                                  nsteps = 200,
                                  use_sparse = TRUE,
                                  parallel = FALSE,
                                  verbose = TRUE) {
    
    if (verbose) {
        cat("=== 高维稀疏TransCox BIC参数选择 ===\n")
        cat("lambda1候选值数量:", length(lambda1_vec), "\n")
        cat("lambda2候选值数量:", length(lambda2_vec), "\n") 
        cat("lambda_beta候选值数量:", length(lambda_beta_vec), "\n")
        cat("总组合数:", length(lambda1_vec) * length(lambda2_vec) * length(lambda_beta_vec), "\n")
    }
    
    # 加载必要的R函数
    if (!exists("GetBIC")) {
        source(file.path(getwd(), "R", "GetBIC.R"))
    }
    if (!exists("GetLogLike")) {
        source(file.path(getwd(), "R", "GetLogLike.R"))
    }
    if (!exists("dQtocumQ")) {
        source(file.path(getwd(), "R", "dQtocumQ.R"))
    }
    
    # 加载Python函数
    if (use_sparse) {
        if (verbose) cat("加载稀疏版本的Python函数...\n")
        source_python(file.path(getwd(), "inst", "python", "TransCoxFunction_Sparse.py"))
    } else {
        source_python(system.file("python", "TransCoxFunction.py", package = "TransCox"))
    }
    
    # 估计源域参数
    if (verbose) cat("估计源域参数...\n")
    if (use_sparse) {
        # 使用稀疏版本
        source(file.path(getwd(), "R", "GetAuxSurv_Sparse.R"))
        Cout <- GetAuxSurv_Sparse(auxData, cov = cov)
    } else {
        Cout <- GetAuxSurv(auxData, cov = cov)
    }
    
    # 计算目标域参数
    Pout <- GetPrimaryParam(primData, q = Cout$q, estR = Cout$estR)
    
    # 准备数据
    CovData = Pout$primData[, cov]
    status = Pout$primData[, statusvar]
    cumH = Pout$primData$fullCumQ
    hazards = Pout$dQ$dQ
    
    # 初始化BIC数组
    BIC_array <- array(NA, dim = c(length(lambda1_vec), length(lambda2_vec), length(lambda_beta_vec)),
                      dimnames = list(lambda1_vec, lambda2_vec, lambda_beta_vec))
    
    # 进度条
    total_combinations <- length(lambda1_vec) * length(lambda2_vec) * length(lambda_beta_vec)
    if (verbose) {
        pb <- txtProgressBar(min = 0, max = total_combinations, style = 3, initial = 0)
    }
    
    combination_count <- 0
    
    # 网格搜索
    for (i in 1:length(lambda1_vec)) {
        for (j in 1:length(lambda2_vec)) {
            for (k in 1:length(lambda_beta_vec)) {
                
                combination_count <- combination_count + 1
                if (verbose) setTxtProgressBar(pb, combination_count)
                
                lambda1 <- lambda1_vec[i]
                lambda2 <- lambda2_vec[j]
                lambda_beta <- lambda_beta_vec[k]
                
                tryCatch({
                    if (use_sparse && lambda_beta > 0) {
                        # 使用稀疏版本
                        result <- TransCox_Sparse(
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
                            verbose = FALSE
                        )
                        
                        eta <- result[[1]]
                        xi <- result[[2]]
                        beta_t <- result[[3]]
                        newBeta <- beta_t  # 直接使用beta_t
                        
                    } else {
                        # 使用原始版本
                        test <- TransCox(
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
                        
                        eta <- test[[1]]
                        xi <- test[[2]]
                        newBeta <- Pout$estR + eta
                    }
                    
                    newHaz <- Pout$dQ$dQ + xi
                    
                    # 计算BIC
                    BICvalue <- GetBIC(
                        status = status,
                        CovData = CovData,
                        hazards = hazards,
                        newBeta = newBeta,
                        newHaz = newHaz,
                        eta = eta,
                        xi = xi,
                        cutoff = 1e-5
                    )
                    
                    BIC_array[i, j, k] <- BICvalue
                    
                }, error = function(e) {
                    if (verbose) {
                        cat("\n警告: lambda组合 (", lambda1, ",", lambda2, ",", lambda_beta, ") 失败:", e$message, "\n")
                    }
                    BIC_array[i, j, k] <- NA
                })
            }
        }
    }
    
    if (verbose) {
        close(pb)
        cat("\n参数搜索完成！\n")
    }
    
    # 找到最优参数
    min_idx <- which(BIC_array == min(BIC_array, na.rm = TRUE), arr.ind = TRUE)
    
    if (nrow(min_idx) > 1) {
        # 如果有多个最优值，选择第一个
        min_idx <- min_idx[1, , drop = FALSE]
    }
    
    best_lambda1 <- lambda1_vec[min_idx[1]]
    best_lambda2 <- lambda2_vec[min_idx[2]]
    best_lambda_beta <- lambda_beta_vec[min_idx[3]]
    
    if (verbose) {
        cat("最优参数:\n")
        cat("  lambda1 (eta):", best_lambda1, "\n")
        cat("  lambda2 (xi):", best_lambda2, "\n")
        cat("  lambda_beta:", best_lambda_beta, "\n")
        cat("  最小BIC:", min(BIC_array, na.rm = TRUE), "\n")
    }
    
    # 计算最优参数下的结果
    if (verbose) cat("使用最优参数计算最终结果...\n")
    
    if (use_sparse && best_lambda_beta > 0) {
        final_result <- TransCox_Sparse(
            CovData = as.matrix(CovData),
            cumH = cumH,
            hazards = hazards,
            status = status,
            estR = Pout$estR,
            Xinn = Pout$Xinn,
            lambda1 = best_lambda1,
            lambda2 = best_lambda2,
            lambda_beta = best_lambda_beta,
            learning_rate = learning_rate,
            nsteps = nsteps,
            verbose = verbose
        )
        
        final_eta <- final_result[[1]]
        final_xi <- final_result[[2]]
        final_beta <- final_result[[3]]
        convergence_info <- final_result[[4]]
        
    } else {
        final_test <- TransCox(
            CovData = as.matrix(CovData),
            cumH = cumH,
            hazards = hazards,
            status = status,
            estR = Pout$estR,
            Xinn = Pout$Xinn,
            lambda1 = best_lambda1,
            lambda2 = best_lambda2,
            learning_rate = learning_rate,
            nsteps = nsteps
        )
        
        final_eta <- final_test[[1]]
        final_xi <- final_test[[2]]
        final_beta <- Pout$estR + final_eta
        convergence_info <- NULL
    }
    
    return(list(
        best_lambda1 = best_lambda1,
        best_lambda2 = best_lambda2,
        best_lambda_beta = best_lambda_beta,
        BIC_array = BIC_array,
        final_eta = final_eta,
        final_xi = final_xi,
        final_beta = final_beta,
        convergence_info = convergence_info,
        source_estR = Pout$estR,
        lambda1_vec = lambda1_vec,
        lambda2_vec = lambda2_vec,
        lambda_beta_vec = lambda_beta_vec
    ))
}

#' 向后兼容的SelParam_By_BIC函数
#' 
SelParam_By_BIC <- function(primData, auxData, cov = c("X1", "X2"),
                           statusvar = "status",
                           lambda1_vec = c(0.1, 0.5, seq(1, 10, by = 0.5)),
                           lambda2_vec = c(0.1, 0.5, seq(1, 10, by = 0.5)),
                           learning_rate = 0.004,
                           nsteps = 100,
                           use_sparse = NULL,
                           ...) {
    
    # 自动检测是否使用稀疏版本
    if (is.null(use_sparse)) {
        n_samples <- nrow(primData)
        n_features <- length(cov)
        use_sparse <- (n_features > n_samples / 2)
        
        if (use_sparse) {
            cat("检测到高维数据，自动使用稀疏版本\n")
        }
    }
    
    if (use_sparse) {
        # 使用新的稀疏版本
        result <- SelParam_By_BIC_Sparse(
            primData = primData,
            auxData = auxData,
            cov = cov,
            statusvar = statusvar,
            lambda1_vec = lambda1_vec,
            lambda2_vec = lambda2_vec,
            learning_rate = learning_rate,
            nsteps = nsteps,
            ...
        )
        
        # 保持原始API格式
        return(list(
            best_la1 = result$best_lambda1,
            best_la2 = result$best_lambda2,
            BICmat = result$BIC_array[,,1]  # 返回lambda_beta=0的切片以保持兼容性
        ))
        
    } else {
        # 使用原始版本
        # [原始代码保持不变]
        TransCox <- NULL
        .onLoad <- function(libname, pkgname) {
            tf <<- reticulate::import("tensorflow", delay_load = TRUE)
            tfp <<- reticulate::import("tensorflow_probability", delay_load = TRUE)
            np <<- reticulate::import("numpy", delay_load = TRUE)
            source_python(system.file("python", "TransCoxFunction.py", package = "TransCox"))
        }
        
        Cout <- GetAuxSurv(auxData, cov = cov)
        Pout <- GetPrimaryParam(primData, q = Cout$q, estR = Cout$estR)
        
        CovData = Pout$primData[, cov]
        status = Pout$primData[, statusvar]
        cumH = Pout$primData$fullCumQ
        hazards = Pout$dQ$dQ
        
        BICmat <- matrix(NA, length(lambda1_vec), length(lambda2_vec))
        pb = txtProgressBar(min = 0, max = (length(lambda1_vec))^2, style = 2, initial = 0)
        for(i in 1:length(lambda1_vec)) {
            for(j in 1:length(lambda2_vec)) {
                
                thisi = (i-1) * length(lambda1_vec) + j
                setTxtProgressBar(pb, thisi)
                
                lambda1 = lambda1_vec[i]
                lambda2 = lambda2_vec[j]
                
                test <- TransCox(CovData = as.matrix(CovData),
                               cumH = cumH,
                               hazards = hazards,
                               status = status,
                               estR = Pout$estR,
                               Xinn = Pout$Xinn,
                               lambda1 = lambda1, lambda2 = lambda2,
                               learning_rate = learning_rate,
                               nsteps = nsteps)
                names(test) <- c("eta", "xi")
                newBeta = Pout$estR + test$eta
                newHaz = Pout$dQ$dQ + test$xi
                
                BICvalue <- GetBIC(status = status,
                                 CovData = CovData,
                                 hazards = hazards,
                                 newBeta = newBeta,
                                 newHaz = newHaz,
                                 eta = test$eta,
                                 xi = test$xi,
                                 cutoff = 1e-5)
                
                BICmat[i,j] <- BICvalue
            }
        }
        close(pb)
        rownames(BICmat) = lambda1_vec
        colnames(BICmat) <- lambda2_vec
        
        idx0 <- which(BICmat == min(BICmat, na.rm = TRUE), arr.ind = TRUE)
        
        b_lambda1 <- lambda1_vec[idx0[1]]
        b_lambda2 <- lambda2_vec[idx0[2]]
        
        return(list(best_la1 = b_lambda1,
                   best_la2 = b_lambda2,
                   BICmat = BICmat))
    }
}