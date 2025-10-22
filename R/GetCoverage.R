#' 计算置信区间覆盖率
#' 
#' @description 
#' 计算参数估计的95%置信区间是否覆盖真实值
#' 
#' @param estimate 参数估计值
#' @param trueBeta 真实参数值
#' @param SE 标准误
#' 
#' @return 0或1，表示是否覆盖
#' @export
GetCoverage <- function(estimate, trueBeta,
                        SE) {
    LL <- estimate - 1.96*SE
    UU <- estimate + 1.96*SE
    return(as.numeric(trueBeta<UU & trueBeta>LL))
}
