#' 将增量风险转换为累积风险
#' 
#' @description 
#' 将增量风险函数dQ转换为累积风险函数cumQ，支持不同的状态编码格式
#' 
#' @param dQ 增量风险向量
#' @param status 状态向量，可选。支持0/1或1/2编码
#' 
#' @return 累积风险向量
#' @export
dQtocumQ <- function(dQ, status = NULL) {
    if(length(status)>0) {
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
        
        cumQ = rep(NA, length(status))
        ncount = 1
        for(i in 1:length(status)) {
            if(status[i] == event_indicator) {
                cumQ[i] = sum(dQ[1:ncount])
                ncount = ncount + 1
                if(ncount > length(dQ)) {
                    ncount = length(dQ)
                }
            } else {
                cumQ[i] = sum(dQ[1:ncount])
            }
        }
    } else {
        cumQ = rep(NA, length(dQ))
        for(i in 1:length(dQ)) {
            cumQ[i] = sum(dQ[1:i])
        }
    }
    return(cumQ)
}
