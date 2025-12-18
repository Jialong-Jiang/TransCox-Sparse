#' Convert Incremental Hazards to Cumulative Hazards
#' 
#' @description 
#' Converts incremental hazard function dQ to cumulative hazard function cumQ, supporting different status encoding formats.
#' 
#' @param dQ Vector of incremental hazards.
#' @param status Vector of status indicators, optional. Supports 0/1 or 1/2 encoding.
#' 
#' @return Vector of cumulative hazards.
#' @export
dQtocumQ <- function(dQ, status = NULL) {
    if(length(status)>0) {
        # Check status encoding format
        if (all(status %in% c(0, 1))) {
            # 0/1 encoding, event is 1
            event_indicator <- 1
        } else if (all(status %in% c(1, 2))) {
            # 1/2 encoding, event is 2
            event_indicator <- 2
        } else {
            stop("Unsupported status encoding. Please use 0/1 or 1/2 encoding.")
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
