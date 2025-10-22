#' 生成高维稀疏生存数据（Weibull）
#'
#' 参照低维GenSimData的Weibull生成策略，扩展到高维稀疏，
#' 在活跃特征上构建主-源域的相关性以支持迁移学习。
#'
#' @param n_main 主数据集样本数
#' @param n_aux 辅助数据集样本数
#' @param n_test 测试数据集样本数
#' @param p 特征维度
#' @param p_active 活跃特征数量
#' @param beta_true 真实系数向量（可选）
#' @param transfer_strength 迁移学习强度（0-1），用于控制主-源域相关性
#' @param noise_level 协变量噪声强度
#' @param censoring_rate 目标删失率（近似控制）
#' @param seed 随机种子
#' @param verbose 是否输出详细信息
#'
#' @return list(main_data, aux_data, test_data, beta_true, active_features, data_info)
#' @export
generate_sparse_survival_data <- function(
  n_main = 200,
  n_aux = 600,
  n_test = 300,
  p = 100,
  p_active = 15,
  beta_true = NULL,
  transfer_strength = 0.8,
  noise_level = 0.1,
  censoring_rate = 0.3,
  seed = NULL,
  verbose = TRUE
) {
  if (!is.null(seed)) set.seed(seed)

  # 真实稀疏beta
  if (is.null(beta_true)) {
    beta_true <- rep(0, p)
    active_indices <- sort(sample(1:p, p_active))
    beta_true[active_indices] <- stats::runif(p_active, 0.3, 0.9) * sample(c(-1,1), p_active, replace = TRUE)
  } else {
    active_indices <- which(beta_true != 0)
  }

  if (verbose) {
    cat("真实活跃特征索引:", active_indices, "\n")
    cat("真实系数值:", round(beta_true[active_indices], 3), "\n")
  }

  # 协变量生成（标准正态）
  X_main <- matrix(rnorm(n_main * p, 0, 1), nrow = n_main, ncol = p)
  X_aux  <- matrix(rnorm(n_aux  * p, 0, 1), nrow = n_aux,  ncol = p)
  X_test <- matrix(rnorm(n_test * p, 0, 1), nrow = n_test, ncol = p)
  colnames(X_main) <- paste0("X", 1:p)
  colnames(X_aux)  <- paste0("X", 1:p)
  colnames(X_test) <- paste0("X", 1:p)

  # 在活跃特征上增强主-源域相关性（rho基于transfer_strength）
  rho <- min(0.35 + 0.5 * transfer_strength, 0.75)  # 0.35~0.75
  for (j in active_indices) {
    idx_resample <- sample(1:n_main, n_aux, replace = TRUE)
    main_signal  <- X_main[idx_resample, j]
    X_aux[, j]   <- rho * main_signal + sqrt(1 - rho^2) * X_aux[, j] + rnorm(n_aux, 0, noise_level * 0.2)
  }

  # 辅助域系数：与主域相近，但存在小差异
  beta_aux <- beta_true * (1 + rnorm(p, 0, 0.05))
  beta_aux[beta_true == 0] <- 0

  # Weibull生存时间生成（参照低维公式）
  gen_weibull <- function(X, beta, shape = 1.6) {
    XB <- as.vector(X %*% beta)
    # 注入风险噪声，降低基线可分性但保留信号
    eps <- rnorm(length(XB), 0, 0.4)
    scale_vec <- exp(-(XB + eps) / 2)
    T <- stats::rweibull(n = length(XB), shape = shape, scale = scale_vec)
    # 近似控制删失率：使用分位数作为上界
    c_upper <- stats::quantile(T, probs = min(0.85, 1 - censoring_rate))
    C <- stats::runif(length(XB), 0, as.numeric(c_upper))
    time   <- ifelse(T < C, T, C)
    status <- ifelse(T < C, 2, 1)  # 2=事件，1=删失（与GenSimData一致）
    list(time = time, status = status)
  }

  # 主域
  surv_main <- gen_weibull(X_main, beta_true, shape = 2)
  main_data <- data.frame(time = surv_main$time, status = surv_main$status, X_main)
  colnames(main_data)[3:ncol(main_data)] <- paste0("X", 1:p)

  # 辅助域
  surv_aux <- gen_weibull(X_aux, beta_aux, shape = 2)
  aux_data <- data.frame(time = surv_aux$time, status = surv_aux$status, X_aux)
  colnames(aux_data)[3:ncol(aux_data)] <- paste0("X", 1:p)

  # 测试集（与主域分布一致但独立）
  surv_test <- gen_weibull(X_test, beta_true, shape = 2)
  test_data <- data.frame(time = surv_test$time, status = surv_test$status, X_test)
  colnames(test_data)[3:ncol(test_data)] <- paste0("X", 1:p)

  if (verbose) {
    cat("数据生成完成!\n")
    cat(sprintf("主域事件率: %.1f%%\n", mean(main_data$status == 2) * 100))
    cat(sprintf("源域事件率: %.1f%%\n", mean(aux_data$status == 2) * 100))
    cat(sprintf("测试事件率: %.1f%%\n", mean(test_data$status == 2) * 100))
  }

  list(
    main_data = main_data,
    aux_data = aux_data,
    test_data = test_data,
    beta_true = beta_true,
    active_features = active_indices,
    data_info = list(
      n_main = n_main,
      n_aux = n_aux,
      n_test = n_test,
      p = p,
      p_active = p_active,
      transfer_strength = transfer_strength,
      rho = rho
    )
  )
}