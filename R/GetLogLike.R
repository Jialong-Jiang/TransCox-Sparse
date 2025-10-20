GetLogLike <- function(status, CovData, hazards,
                       newBeta,
                       newHaz) {
    XB = as.matrix(CovData) %*% newBeta
    newCumH <- dQtocumQ(newHaz, status)
    
    # 防止log函数出现数值问题
    # 确保newHaz中的值都是正数，避免log(0)或log(负数)
    newHaz_safe <- pmax(newHaz, 1e-10)  # 将小于1e-10的值设为1e-10
    
    LogL <- sum(XB[status == 2] + log(newHaz_safe)) - sum(newCumH * exp(XB))
    return(LogL)
}
