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
    Logl <- GetLogLike(status = status,
                       CovData = CovData,
                       hazards = hazards,
                       newBeta = newBeta,
                       newHaz = newHaz)
    
    # 自适应阈值（与惩罚参数规模相关），避免梯度法导致“近零但不为零”的系数被误计入
    cutoff_eta <- if (!is.null(lambda1)) max(cutoff, 0.1 * lambda1) else cutoff
    cutoff_xi  <- if (!is.null(lambda2)) max(cutoff, 0.1 * lambda2) else cutoff
    cutoff_beta <- if (!is.null(lambda_beta)) max(cutoff, 0.1 * lambda_beta) else cutoff

    # 计算有效参数数量
    K_eta <- sum(abs(eta) > cutoff_eta)
    K_xi  <- sum(abs(xi)  > cutoff_xi)
    K_beta <- if (!is.null(newBeta)) sum(abs(newBeta) > cutoff_beta) else 0
    K_total <- K_eta + K_xi + K_beta
    
    n <- length(status)
    
    # 标准BIC（无额外penalty系数）：维度罚项 + 拟合优度
    BIC <- K_total * log(n) - 2 * Logl
    
    return(BIC)
}
