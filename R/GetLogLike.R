#' 计算对数似然函数
#' 
#' @description 
#' 计算TransCox模型的对数似然函数值，支持不同的状态编码格式
#' 
#' @param status 事件状态向量，支持0/1或1/2编码
#' @param CovData 协变量数据矩阵
#' @param hazards 原始风险函数向量
#' @param newBeta 新的回归系数向量
#' @param newHaz 新的风险函数向量
#' 
#' @return 对数似然值
#' @export
GetLogLike <- function(status, CovData, hazards,
                       newBeta,
                       newHaz) {
    XB = as.matrix(CovData) %*% newBeta
    newCumH <- dQtocumQ(newHaz, status)
    
    # 防止log函数出现数值问题
    # 确保newHaz中的值都是正数，避免log(0)或log(负数)
    newHaz_safe <- pmax(newHaz, 1e-10)  # 将小于1e-10的值设为1e-10
    
    # 检测状态编码格式
    if (all(status %in% c(0, 1))) {
        # 0/1编码，事件为1
        event_indicator <- 1
    } else if (all(status %in% c(1, 2))) {
        # 1/2编码，事件为2
        event_indicator <- 2
    } else {
        stop("不支持的状态编码，请使用0/1或1/2编码")
    }
    
    LogL <- sum(XB[status == event_indicator] + log(newHaz_safe)) - sum(newCumH * exp(XB))
    return(LogL)
}
