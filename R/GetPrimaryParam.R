#' 计算目标域特定参数
#' 
#' @description 
#' 计算目标域数据特定的断点和风险率，用于TransCox模型的目标域参数估计
#' 
#' @param primData 目标域数据，包含time和status列
#' @param q 从源域估计得到的参数向量
#' @param estR 从源域估计得到的回归系数
#' 
#' @return 包含目标域特定参数的列表
#' @export
GetPrimaryParam <- function(primData, q, estR) {
    ### get primary data-specific breakpoints and hazards
    primData <- primData[order(primData$time), ]
    dQ <- deltaQ(primData, q)
    primData <- primData[order(primData$time), ]
    fullcum <- rep(0, nrow(primData))
    idx0 <- match(dQ$t, primData$time)

    ### get cumQ
    fullcum[idx0] <- dQ$cumQ_upd
    for(i in 1:length(fullcum)) {
        if(fullcum[i] == 0 & i>1) {
            fullcum[i] = fullcum[i-1]
        }
    }
    primData$fullCumQ = fullcum

    #### get Ximat - 支持0/1和1/2两种编码
    # 检测状态编码格式
    if (all(primData$status %in% c(0, 1))) {
        # 0/1编码，事件为1
        event_indicator <- 1
    } else if (all(primData$status %in% c(1, 2))) {
        # 1/2编码，事件为2
        event_indicator <- 2
    } else {
        stop("不支持的状态编码，请使用0/1或1/2编码")
    }
    
    Ximat <- matrix(0, nrow(primData), nrow(primData))
    for(i in 1:nrow(primData)) {
        tmpidx = rep(0, nrow(primData))
        for(j in 1:i) {
            if(primData$status[j] == event_indicator)
                tmpidx[j] = 1
        }
        Ximat[i,] <- tmpidx
    }
    Xinn <- Ximat[,which(primData$status == event_indicator)]
    return(list(primData = primData,
                Xinn = Xinn,
                dQ = dQ,
                estR = estR))
}
