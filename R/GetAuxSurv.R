#' 标准Cox回归估计源域参数
#' 
#' @description 
#' 使用标准Cox回归模型估计源域参数，包括回归系数和基线风险函数
#' 
#' @param auxData 源域数据，包含time、status和协变量列
#' @param weights 样本权重，可选
#' @param cov 协变量名称向量
#' 
#' @return 包含estR（回归系数）和q（基线风险参数）的列表
#' @export
GetAuxSurv <- function(auxData, weights = NULL, cov = c("X1", "X2")) {
    res.cox <- c()
    functext = paste0("res.cox <- survival::coxph(survival::Surv(time, status == 2) ~ ", paste(cov, collapse = "+"), ", data = auxData, weights = weights)")
    ### Give auxiliary data, estimate r and q
    eval(parse(text = functext))
    bhest <- survival::basehaz(res.cox, centered=FALSE) ## get baseline cumulative hazards
    estR <- res.cox$coefficients
    q <- data.frame(cumHazards = bhest$hazard,
                    breakPoints = bhest$time)
    return(list(estR = estR,
                q = q))
}
