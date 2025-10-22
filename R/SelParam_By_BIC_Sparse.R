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
# 辅助函数：生成细搜索向量
generate_fine_search_vector <- function(best_value, coarse_vec) {
    # 找到最优值在粗搜索向量中的位置
    idx <- which.min(abs(coarse_vec - best_value))
    
    # 确定搜索范围
    if (idx == 1) {
        # 最优值在最小端
        lower <- best_value
        upper <- if (length(coarse_vec) > 1) coarse_vec[2] else best_value * 2
    } else if (idx == length(coarse_vec)) {
        # 最优值在最大端
        lower <- coarse_vec[idx - 1]
        upper <- best_value
    } else {
        # 最优值在中间
        lower <- coarse_vec[idx - 1]
        upper <- coarse_vec[idx + 1]
    }
    
    # 在范围内生成3-5个细搜索点
    if (best_value == 0) {
        # 对于lambda_beta=0的特殊情况
        fine_vec <- c(0, seq(0.0001, upper, length.out = 3))
    } else {
        # 在[lower, upper]范围内生成均匀分布的点
        fine_vec <- seq(lower, upper, length.out = 5)
    }
    
    # 移除重复值并排序
    fine_vec <- sort(unique(fine_vec))
    return(fine_vec)
}

SelParam_By_BIC_Sparse <- function(primData, auxData, cov = c("X1", "X2"),
                                  statusvar = "status",
                                  lambda1_vec = c(0.1, 0.5, 1.0, 2.0),  # 粗搜索：4个值
                                  lambda2_vec = c(0.1, 0.5, 1.0, 2.0),  # 粗搜索：4个值
                                  lambda_beta_vec = c(0, 1e-3, 1e-2, 5e-2),  # 粗搜索：4个值
                                  learning_rate = 0.004,
                                  nsteps = 200,
                                  use_sparse = TRUE,
                                  parallel = FALSE,
                                  verbose = TRUE,
                                  n_cores = NULL) {
    
    # BIC参数选择开始
    if (verbose) cat("🔍 开始BIC参数选择...\n")
    
    # 创建结果缓存环境
    result_cache <- new.env()
    
    # 并行计算初始化
    if (parallel) {
        # 检测可用核心数
        if (is.null(n_cores)) {
            n_cores <- max(1, parallel::detectCores() - 1)  # 保留一个核心
        }
        n_cores <- min(n_cores, parallel::detectCores())
        
        if (verbose) cat(sprintf("🚀 启用并行计算，使用 %d 个核心\n", n_cores))
        
        # 检查并加载必要的包
        if (!requireNamespace("parallel", quietly = TRUE)) {
            if (verbose) cat("⚠️  parallel包不可用，切换到串行模式\n")
            parallel <- FALSE
        } else if (!requireNamespace("doParallel", quietly = TRUE)) {
            if (verbose) cat("⚠️  doParallel包不可用，切换到串行模式\n")
            parallel <- FALSE
        } else {
            # 设置并行后端
            cl <- parallel::makeCluster(n_cores)
            doParallel::registerDoParallel(cl)
            
            # 确保在函数退出时清理集群
            on.exit({
                if (exists("cl") && !is.null(cl)) {
                    parallel::stopCluster(cl)
                    if (verbose) cat("🔄 并行集群已清理\n")
                }
            }, add = TRUE)
        }
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
    
    # 预计算常用数据，减少重复计算和优化数据传输
    if (verbose) cat("📊 预计算数据以优化性能...\n")
    
    # 预计算协变量矩阵，优化内存布局
    CovData_precomputed <- as.matrix(Pout$primData[, cov, drop = FALSE], mode = "double")
    storage.mode(CovData_precomputed) <- "double"
    status_precomputed <- as.integer(Pout$primData[, statusvar])
    
    # 预计算源域和目标域参数
    estR_precomputed <- as.double(Pout$estR)
    Xinn_precomputed <- as.matrix(Pout$Xinn, mode = "double")
    storage.mode(Xinn_precomputed) <- "double"
    cumH_precomputed <- as.double(Pout$primData$fullCumQ)
    hazards_precomputed <- as.double(Pout$dQ$dQ)
    
    # 创建优化的数据包，一次性传输到Python环境
    data_package <- list(
        CovData = CovData_precomputed,
        cumH = cumH_precomputed,
        hazards = hazards_precomputed,
        status = status_precomputed,
        estR = estR_precomputed,
        Xinn = Xinn_precomputed
    )
    
    # 准备数据（保持向后兼容）
    CovData = Pout$primData[, cov]
    status = Pout$primData[, statusvar]
    cumH = Pout$primData$fullCumQ
    hazards = Pout$dQ$dQ
    
    # 初始化BIC数组
    BIC_array <- array(NA, dim = c(length(lambda1_vec), length(lambda2_vec), length(lambda_beta_vec)),
                      dimnames = list(lambda1_vec, lambda2_vec, lambda_beta_vec))
    
    # 性能优化：预计算常用数据
    CovData_matrix <- CovData_precomputed
    hazards_base <- hazards_precomputed
    
    # 分层搜索策略：先粗搜索，再细搜索
    if (verbose) cat("🔍 开始分层参数搜索...\n")
    
    # 第一阶段：粗搜索（使用当前的参数向量）
    coarse_grid <- expand.grid(
        lambda1 = lambda1_vec,
        lambda2 = lambda2_vec,
        lambda_beta = lambda_beta_vec,
        stringsAsFactors = FALSE
    )
    
    if (verbose) {
        cat(sprintf("📊 第一阶段：粗搜索，共%d个参数组合...\n", nrow(coarse_grid)))
    }
    
    # 使用粗搜索网格
    param_grid <- coarse_grid
    
    total_combinations <- nrow(param_grid)
    if (verbose) cat(sprintf("📊 总共需要评估 %d 个参数组合\n", total_combinations))
    
    # 智能缓存机制：避免重复计算相似参数组合
    cache_tolerance <- 1e-6  # 参数相似性容忍度
    bic_cache <- list()  # BIC结果缓存
    
    # 缓存查找函数
    find_cached_result <- function(lambda1, lambda2, lambda_beta) {
        for (cached in bic_cache) {
            if (abs(cached$lambda1 - lambda1) < cache_tolerance &&
                abs(cached$lambda2 - lambda2) < cache_tolerance &&
                abs(cached$lambda_beta - lambda_beta) < cache_tolerance) {
                return(cached)
            }
        }
        return(NULL)
    }
    
    # 定义单个参数组合的评估函数（带早停机制和缓存）
    evaluate_params <- function(idx, param_row, data_pkg, use_sparse_flag, lr, n_steps, verb = FALSE) {
        lambda1 <- param_row$lambda1
        lambda2 <- param_row$lambda2
        lambda_beta <- param_row$lambda_beta
        
        # 缓存检查：查找是否已计算过相似参数组合
        cached_result <- find_cached_result(lambda1, lambda2, lambda_beta)
        if (!is.null(cached_result)) {
            return(list(
                idx = idx, lambda1 = lambda1, lambda2 = lambda2, lambda_beta = lambda_beta,
                bic = cached_result$bic, success = TRUE, cached = TRUE
            ))
        }
        
        # 早停机制：快速检测明显无效的参数组合
        if (lambda1 <= 0 || lambda2 <= 0 || (use_sparse_flag && lambda_beta < 0)) {
            return(list(
                lambda1 = lambda1, lambda2 = lambda2, lambda_beta = lambda_beta,
                bic = Inf, success = FALSE, early_stop = TRUE
            ))
        }
        
        tryCatch({
            if (use_sparse_flag && !is.null(lambda_beta) && lambda_beta > 0) {
                # 使用稀疏版本
                result <- TransCox_Sparse(
                    CovData = data_pkg$CovData,
                    cumH = data_pkg$cumH,
                    hazards = data_pkg$hazards,
                    status = data_pkg$status,
                    estR = data_pkg$estR,
                    Xinn = data_pkg$Xinn,
                    lambda1 = lambda1,
                    lambda2 = lambda2,
                    lambda_beta = lambda_beta,
                    learning_rate = lr,
                    nsteps = n_steps,
                    verbose = FALSE
                )
                
                eta <- result[[1]]
                xi <- result[[2]]
                newBeta <- result[[3]]
                
            } else {
                # 使用原始版本
                test <- TransCox(
                    CovData = data_pkg$CovData,
                    cumH = data_pkg$cumH,
                    hazards = data_pkg$hazards,
                    status = data_pkg$status,
                    estR = data_pkg$estR,
                    Xinn = data_pkg$Xinn,
                    lambda1 = lambda1,
                    lambda2 = lambda2,
                    learning_rate = lr,
                    nsteps = n_steps
                )
                
                eta <- test[[1]]
                xi <- test[[2]]
                newBeta <- data_pkg$estR + eta
            }
            
            newHaz <- data_pkg$hazards + xi
            
            # 早停机制：检查结果的有效性
            if (any(is.na(eta)) || any(is.na(xi)) || any(is.na(newBeta)) || 
                any(is.infinite(eta)) || any(is.infinite(xi)) || any(is.infinite(newBeta))) {
                return(list(
                    idx = idx, lambda1 = lambda1, lambda2 = lambda2, lambda_beta = lambda_beta,
                    bic = Inf, success = FALSE, early_stop = TRUE,
                    error = "Invalid results (NA or Inf values)"
                ))
            }
            
            # 计算BIC - 使用data_pkg中的数据确保并行计算兼容性
            bic_value <- GetBIC(
                status = data_pkg$status,
                CovData = data_pkg$CovData,
                hazards = data_pkg$hazards,
                newBeta = newBeta,
                newHaz = newHaz,
                eta = eta,
                xi = xi,
                cutoff = 1e-5,
                lambda1 = lambda1,
                lambda2 = lambda2,
                lambda_beta = lambda_beta
            )
            
            # 早停机制：检查BIC值的有效性
            if (is.na(bic_value) || is.infinite(bic_value) || bic_value > 1e6) {
                return(list(
                    idx = idx, lambda1 = lambda1, lambda2 = lambda2, lambda_beta = lambda_beta,
                    bic = Inf, success = FALSE, early_stop = TRUE,
                    error = "Invalid BIC value"
                ))
            }
             
             # 存储到缓存
             bic_cache <<- append(bic_cache, list(list(
                 lambda1 = lambda1, lambda2 = lambda2, lambda_beta = lambda_beta,
                 bic = bic_value
             )))
             
             return(list(
                 idx = idx,
                 lambda1 = lambda1,
                 lambda2 = lambda2,
                 lambda_beta = lambda_beta,
                 bic = bic_value,
                 eta = eta,
                 xi = xi,
                 newBeta = newBeta,
                 success = TRUE
             ))
            
        }, error = function(e) {
            return(list(
                idx = idx,
                lambda1 = lambda1,
                lambda2 = lambda2,
                lambda_beta = lambda_beta,
                bic = Inf,
                eta = NULL,
                xi = NULL,
                newBeta = NULL,
                success = FALSE,
                error = e$message
            ))
        })
    }
    
    # 执行参数搜索（并行或串行）
    results <- NULL
    parallel_success <- FALSE
    
    if (parallel && exists("cl")) {
        if (verbose) cat("🚀 开始并行参数搜索...\n")
        
        # 尝试并行计算
        tryCatch({
            # 加载必要的包到工作进程
            parallel::clusterEvalQ(cl, {
                library(reticulate)
                # 重新加载Python函数
                if (file.exists("inst/python/TransCoxFunction_Sparse.py")) {
                    reticulate::source_python("inst/python/TransCoxFunction_Sparse.py")
                }
                if (file.exists("inst/python/TransCoxFunction.py")) {
                    reticulate::source_python("inst/python/TransCoxFunction.py")
                }
            })
            
            # 导出R函数
            parallel::clusterExport(cl, c("GetBIC", "evaluate_params"), envir = environment())
            
            # 并行执行
            results <- parallel::parLapply(cl, 1:nrow(param_grid), function(i) {
                evaluate_params(i, param_grid[i, ], data_package, use_sparse, learning_rate, nsteps, FALSE)
            })
            
            parallel_success <- TRUE
            
        }, error = function(e) {
            if (verbose) cat("⚠️  并行计算出错，切换到串行模式:", e$message, "\n")
            results <<- NULL
        })
    }
    
    # 如果并行失败或未启用并行，使用串行模式
    if (!parallel_success || is.null(results)) {
        if (verbose) cat("🔄 开始串行参数搜索...\n")
        
        # 串行执行
        results <- lapply(1:nrow(param_grid), function(i) {
            result <- evaluate_params(i, param_grid[i, ], data_package, use_sparse, learning_rate, nsteps, FALSE)
            
            # 进度报告
            if (verbose && i %% 5 == 0) {
                progress <- i / nrow(param_grid) * 100
                cat(sprintf("📈 进度: %.1f%% (%d/%d)\n", progress, i, nrow(param_grid)))
            }
            
            return(result)
        })
    }
    
    # 处理结果
    BIC_values <- sapply(results, function(x) x$bic)
    successful_results <- results[sapply(results, function(x) x$success)]
    
    if (verbose) {
        success_rate <- length(successful_results) / length(results) * 100
        cat(sprintf("✅ 成功率: %.1f%% (%d/%d)\n", success_rate, length(successful_results), length(results)))
    }
    
    # 重建BIC数组
    BIC_array <- array(Inf, dim = c(length(lambda1_vec), length(lambda2_vec), length(lambda_beta_vec)),
                      dimnames = list(lambda1_vec, lambda2_vec, lambda_beta_vec))
    
    for (result in results) {
        i <- which(lambda1_vec == result$lambda1)
        j <- which(lambda2_vec == result$lambda2)
        k <- which(lambda_beta_vec == result$lambda_beta)
        BIC_array[i, j, k] <- result$bic
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
    current_best_bic <- min(BIC_array, na.rm = TRUE)
    
    if (verbose) {
        cat(sprintf("✅ 粗搜索完成，最优参数: λ1=%.3f, λ2=%.3f, λβ=%.4f, BIC=%.3f\n", 
                   best_lambda1, best_lambda2, best_lambda_beta, current_best_bic))
    }
    
    # 第二阶段：细搜索 - 在最优参数周围进行精细搜索
    if (verbose) cat("📍 第二阶段：在最优区域进行细搜索...\n")
    
    # 生成细搜索的参数向量
    fine_lambda1_vec <- generate_fine_search_vector(best_lambda1, lambda1_vec)
    fine_lambda2_vec <- generate_fine_search_vector(best_lambda2, lambda2_vec)
    fine_lambda_beta_vec <- generate_fine_search_vector(best_lambda_beta, lambda_beta_vec)
    
    # 创建细搜索网格
    fine_grid <- expand.grid(
        lambda1 = fine_lambda1_vec,
        lambda2 = fine_lambda2_vec,
        lambda_beta = fine_lambda_beta_vec,
        stringsAsFactors = FALSE
    )
    
    # 移除已经计算过的组合
    fine_grid <- fine_grid[!paste(fine_grid$lambda1, fine_grid$lambda2, fine_grid$lambda_beta) %in% 
                          paste(coarse_grid$lambda1, coarse_grid$lambda2, coarse_grid$lambda_beta), ]
    
    if (nrow(fine_grid) > 0 && verbose) {
        cat(sprintf("🔬 细搜索新增%d个参数组合...\n", nrow(fine_grid)))
        
        # 执行细搜索
        fine_results <- lapply(1:nrow(fine_grid), function(i) {
            result <- evaluate_params(i, fine_grid[i, ], data_package, use_sparse, learning_rate, nsteps, FALSE)
            
            # 进度报告
            if (verbose && i %% 3 == 0) {
                progress <- i / nrow(fine_grid) * 100
                cat(sprintf("🔬 细搜索进度: %.1f%% (%d/%d)\n", progress, i, nrow(fine_grid)))
            }
            
            return(result)
        })
        
        # 合并粗搜索和细搜索结果
        all_results <- c(results, fine_results)
        
        # 更新最优参数
        fine_BIC_values <- sapply(fine_results, function(x) x$bic)
        min_fine_bic <- min(fine_BIC_values, na.rm = TRUE)
        
        if (min_fine_bic < current_best_bic) {
            min_fine_idx <- which.min(fine_BIC_values)
            best_fine_result <- fine_results[[min_fine_idx]]
            
            best_lambda1 <- best_fine_result$lambda1
            best_lambda2 <- best_fine_result$lambda2
            best_lambda_beta <- best_fine_result$lambda_beta
            current_best_bic <- min_fine_bic
            
            if (verbose) {
                cat(sprintf("🎯 细搜索找到更优参数: λ1=%.3f, λ2=%.3f, λβ=%.4f, BIC=%.3f\n", 
                           best_lambda1, best_lambda2, best_lambda_beta, current_best_bic))
            }
        }
        
        # 更新结果列表
        results <- all_results
    }
    
    # 跳过局部微调，因为细搜索已经提供了足够的精度
    if (verbose) cat("✅ 分层搜索完成，跳过额外的局部微调\n")
    
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