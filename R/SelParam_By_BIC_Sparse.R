#' BIC-Based Parameter Selection for Sparse TransCox
#' 
#' @description
#' Performs Bayesian Information Criterion (BIC) based parameter selection for
#' the sparse TransCox model. This function extends the original BIC selection
#' to support three-dimensional parameter tuning including lambda_beta for
#' target domain sparsity control.
#' 
#' @details
#' This function conducts a comprehensive grid search over three regularization
#' parameters (lambda1, lambda2, lambda_beta) to find the optimal combination
#' that minimizes the BIC. The search can be parallelized for computational
#' efficiency when dealing with large parameter grids.
#' 
#' @param primData A data.frame containing the target domain survival data.
#' @param auxData A data.frame containing the source domain survival data.
#' @param cov A character vector specifying the names of covariates to be used.
#'   Default is c("X1", "X2").
#' @param statusvar A character string specifying the name of the event status
#'   variable. Default is "status".
#' @param lambda1_vec Numeric vector of candidate values for lambda1 (L1 penalty
#'   for eta parameter). Default provides a reasonable range.
#' @param lambda2_vec Numeric vector of candidate values for lambda2 (L1 penalty
#'   for xi parameter). Default provides a reasonable range.
#' @param lambda_beta_vec Numeric vector of candidate values for lambda_beta
#'   (L1 penalty for beta_t parameter). Default includes zero for non-sparse option.
#' @param learning_rate Numeric. Learning rate for the optimization algorithm.
#'   Default is 0.004.
#' @param nsteps Integer. Maximum number of optimization steps. Default is 200.
#' @param use_sparse Logical. Whether to use sparse implementation. Default is TRUE.
#' @param parallel Logical. Whether to use parallel computation for grid search.
#'   Default is FALSE.
#' @param verbose Logical. Whether to display detailed progress information.
#'   Default is TRUE.
#' 
#' @return A list containing the following components:
#' \describe{
#'   \item{optimal_lambda1}{The optimal lambda1 value that minimizes BIC}
#'   \item{optimal_lambda2}{The optimal lambda2 value that minimizes BIC}
#'   \item{optimal_lambda_beta}{The optimal lambda_beta value that minimizes BIC}
#'   \item{min_bic}{The minimum BIC value achieved}
#'   \item{bic_matrix}{3D array of BIC values for all parameter combinations}
#'   \item{parameter_grid}{Data frame of all tested parameter combinations}
#'   \item{convergence_info}{Information about optimization convergence}
#' }
#' 
#' @examples
#' \dontrun{
#' # Generate example data
#' data <- generate_sparse_survival_data(n_main = 100, n_aux = 200, p = 50)
#' 
#' # Perform BIC-based parameter selection
#' bic_result <- SelParam_By_BIC_Sparse(
#'   primData = data$prim_data,
#'   auxData = data$aux_data,
#'   cov = paste0("X", 1:50),
#'   lambda1_vec = c(0.1, 0.5, 1.0),
#'   lambda2_vec = c(0.1, 0.5, 1.0),
#'   lambda_beta_vec = c(0, 0.001, 0.01)
#' )
#' 
#' # View optimal parameters
#' print(bic_result$optimal_lambda1)
#' print(bic_result$optimal_lambda2)
#' print(bic_result$optimal_lambda_beta)
#' }
#' 
#' @export
#' 
SelParam_By_BIC_Sparse <- function(primData, auxData, cov = c("X1", "X2"),
                                  statusvar = "status",
                                  lambda1_vec = c(0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0),
                                  lambda2_vec = c(0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0),
                                  lambda_beta_vec = c(0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2),
                                  learning_rate = 0.004,
                                  nsteps = 200,
                                  use_sparse = TRUE,
                                  parallel = FALSE,
                                  verbose = TRUE) {
    
    # BIC参数选择开始
    
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
    if (!exists("GetPrimaryParam")) {
        source(file.path(getwd(), "R", "GetPrimaryParam.R"))
    }
    
    # 加载Python函数
    if (use_sparse) {
        if (!exists("TransCox_Sparse")) {
            reticulate::source_python(file.path(getwd(), "inst", "python", "TransCoxFunction_Sparse.py"))
        }
    } else {
        if (!exists("TransCox")) {
            reticulate::source_python(system.file("python", "TransCoxFunction.py", package = "TransCox"))
        }
    }
    
    # 估计源域参数
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
                    if (use_sparse && !is.null(lambda_beta) && length(lambda_beta) == 1 && lambda_beta > 0) {
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
                    BICvalue <- GetBIC(status = status,
                                 CovData = CovData,
                                 hazards = hazards,
                                 newBeta = newBeta,
                                 newHaz = newHaz,
                                 eta = eta,
                                 xi = xi,
                                 cutoff = 1e-5,
                                 lambda1 = lambda1,
                                 lambda2 = lambda2,
                                 lambda_beta = lambda_beta
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
    
    # 二阶段局部细化搜索：只微调lambda_beta以降低计算成本
    K <- min(3, length(lambda1_vec) * length(lambda2_vec) * length(lambda_beta_vec))
    bic_vals <- as.vector(BIC_array)
    ord <- order(bic_vals, na.last = NA)
    top_idx <- ord[1:K]
    idx_mat <- arrayInd(top_idx, .dim = dim(BIC_array))
    
    current_best_bic <- min(bic_vals, na.rm = TRUE)
    
    for (r in 1:nrow(idx_mat)) {
        i0 <- idx_mat[r, 1]; j0 <- idx_mat[r, 2]; k0 <- idx_mat[r, 3]
        l1 <- lambda1_vec[i0]; l2 <- lambda2_vec[j0]; lb_0 <- lambda_beta_vec[k0]
        
        # 构造lambda_beta的局部候选（对数尺度+邻域）
        neighbor_lower <- lambda_beta_vec[max(1, k0 - 1)]
        neighbor_upper <- lambda_beta_vec[min(length(lambda_beta_vec), k0 + 1)]
        scale_set_beta <- c(0.8, 1.0, 1.25)
        lb_scaled <- lb_0 * scale_set_beta
        lb_candidates <- sort(unique(pmax(0, c(lb_0, neighbor_lower, neighbor_upper, lb_scaled))))
        
        for (lb in lb_candidates) {
            if (use_sparse && !is.null(lb) && lb > 0) {
                res <- try(TransCox_Sparse(
                    CovData = as.matrix(CovData),
                    cumH = cumH,
                    hazards = hazards,
                    status = status,
                    estR = Pout$estR,
                    Xinn = Pout$Xinn,
                    lambda1 = l1,
                    lambda2 = l2,
                    lambda_beta = lb,
                    learning_rate = learning_rate,
                    nsteps = nsteps,
                    verbose = FALSE
                ), silent = TRUE)
                if (inherits(res, "try-error")) next
                eta_loc <- res[[1]]; xi_loc <- res[[2]]; beta_loc <- res[[3]]
                newBeta_loc <- beta_loc
            } else {
                # 当lb==0时，使用原始版本（不加beta_t稀疏惩罚）
                test <- TransCox(
                    CovData = as.matrix(CovData),
                    cumH = cumH,
                    hazards = hazards,
                    status = status,
                    estR = Pout$estR,
                    Xinn = Pout$Xinn,
                    lambda1 = l1,
                    lambda2 = l2,
                    learning_rate = learning_rate,
                    nsteps = nsteps
                )
                eta_loc <- test[[1]]; xi_loc <- test[[2]]
                newBeta_loc <- Pout$estR + eta_loc
            }
            
            newHaz_loc <- Pout$dQ$dQ + xi_loc
            bic_loc <- GetBIC(status = status,
                              CovData = CovData,
                              hazards = hazards,
                              newBeta = newBeta_loc,
                              newHaz = newHaz_loc,
                              eta = eta_loc,
                              xi = xi_loc,
                              cutoff = 1e-5,
                              lambda1 = l1,
                              lambda2 = l2,
                              lambda_beta = lb)
            
            if (!is.na(bic_loc) && bic_loc < current_best_bic) {
                current_best_bic <- bic_loc
                best_lambda1 <- l1
                best_lambda2 <- l2
                best_lambda_beta <- lb
            }
        }
    }
    
    # 计算最优参数下的结果
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
        
        # 自动选择稀疏版本
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
        if (!exists("TransCox")) {
            reticulate::source_python(system.file("python", "TransCoxFunction.py", package = "TransCox"))
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
                                 cutoff = 1e-5,
                                 lambda1 = lambda1_vec[i],
                                 lambda2 = lambda2_vec[j],
                                 lambda_beta = NULL)
                
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