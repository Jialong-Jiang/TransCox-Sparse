#' 计算贝叶斯信息准则(BIC)
#' 
#' @description 
#' 计算TransCox模型的BIC值，用于模型选择和参数调优
#' 
#' @param status 事件状态向量
#' @param CovData 协变量数据矩阵
#' @param hazards 风险函数向量
#' @param newBeta 新的回归系数
#' @param newHaz 新的风险函数
#' @param eta eta参数向量
#' @param xi xi参数向量
#' @param cutoff 参数非零判断阈值
#' @param lambda1 eta的L1惩罚参数（用于自适应阈值）
#' @param lambda2 xi的L1惩罚参数（用于自适应阈值）
#' @param lambda_beta beta_t的L1惩罚参数（用于自适应阈值）
#' 
#' @return BIC值
#' @export
GetBIC <- function(status, CovData, hazards,
                   newBeta,
                   newHaz,
                   eta,
                   xi,
                   cutoff = 1e-5,
                   lambda1 = NULL,
                   lambda2 = NULL,
                   lambda_beta = NULL) {
    
    # 性能优化：预计算常用值
    n <- length(status)
    log_n <- log(n)
    
    # 快速计算对数似然
    Logl <- GetLogLike(status = status,
                       CovData = CovData,
                       hazards = hazards,
                       newBeta = newBeta,
                       newHaz = newHaz)
    
    # 检查对数似然的有效性
    if (is.na(Logl) || is.infinite(Logl)) {
        return(Inf)  # 返回无穷大BIC表示无效模型
    }
    
    # 自适应阈值优化：使用向量化操作
    cutoff_eta <- if (!is.null(lambda1)) max(cutoff, 0.1 * lambda1) else cutoff
    cutoff_xi  <- if (!is.null(lambda2)) max(cutoff, 0.1 * lambda2) else cutoff
    cutoff_beta <- if (!is.null(lambda_beta)) max(cutoff, 0.1 * lambda_beta) else cutoff

    # 向量化计算有效参数数量
    K_eta <- sum(abs(eta) > cutoff_eta, na.rm = TRUE)
    K_xi  <- sum(abs(xi) > cutoff_xi, na.rm = TRUE)
    K_beta <- if (!is.null(newBeta)) sum(abs(newBeta) > cutoff_beta, na.rm = TRUE) else 0
    K_total <- K_eta + K_xi + K_beta
    
    # 数值稳定性检查
    if (K_total < 0 || K_total > n) {
        return(Inf)  # 参数数量异常
    }
    
    # 优化的BIC计算：避免重复计算log(n)
    BIC <- K_total * log_n - 2 * Logl
    
    # 最终数值检查
    if (is.na(BIC) || is.infinite(BIC)) {
        return(Inf)
    }
    
    return(BIC)
}
