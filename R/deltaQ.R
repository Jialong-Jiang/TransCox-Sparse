#' Calculate Target Domain Cumulative Hazards
#' 
#' @description 
#' Calculates the cumulative hazard function for target domain data based on parameters estimated from the source domain.
#' 
#' @param primData Target domain data, must contain 'time' and 'status' columns.
#' @param q Hazard parameters estimated from source domain, containing 'breakPoints' and 'cumHazards'.
#' 
#' @return A data frame containing dQ, cumQ, and t.
#' @export
deltaQ <- function(primData, q) {
    #### get cumulative hazards for primary data
    #### from the combined data
    # Check status encoding format and get event data
    if (all(primData$status %in% c(0, 1))) {
        # 0/1 encoding, event is 1
        obsData <- primData[primData$status == 1, ]
    } else if (all(primData$status %in% c(1, 2))) {
        # 1/2 encoding, event is 2
        obsData <- primData[primData$status == 2, ]
    } else {
        stop("Unsupported status encoding. Please use 0/1 or 1/2 encoding.")
    }
    obsData <- obsData[order(obsData$time), ]
    newQ <- data.frame(dQ  = rep(NA, nrow(obsData)),
                       cumQ = rep(NA, nrow(obsData)),
                       t = rep(NA, nrow(obsData)))
    for(i in 1:nrow(obsData)) {
        if (i == 1) {
            newQ$t[i] <- obsData$time[i]
            tmp1 <- q$breakPoints<=obsData$time[i]
            if(all(!tmp1)) {
                idx0 <- 1
            } else {
                idx0 <- max(which(tmp1))
            }
            newQ$dQ[i] = newQ$cumQ[i] <- q$cumHazards[idx0]
        } else {
            newQ$t[i] <- obsData$time[i]
            tmp1 <- q$breakPoints<=obsData$time[i]
            if(all(!tmp1)) {
                idx0 <- 1
            } else {
                idx0 <- max(which(tmp1))
            }
            newQ$cumQ[i] <- q$cumHazards[idx0]
            newQ$dQ[i] <- q$cumHazards[idx0] - newQ$cumQ[(i-1)]
        }
    }
    newQ$dQ[newQ$dQ == 0] <- 0.0001
    newQ$cumQ_upd <- newQ$cumQ
    for(i in 1:nrow(newQ)) {
        newQ$cumQ_upd[i] <- sum(newQ$dQ[1:i])
    }
    return(newQ)
}
