#' Calculate Target Domain Specific Parameters
#' 
#' @description 
#' Calculates target domain specific breakpoints and hazards for TransCox model parameter estimation.
#' 
#' @param primData Target domain data, must contain 'time' and 'status' columns.
#' @param q Parameter vector estimated from source domain.
#' @param estR Regression coefficients estimated from source domain.
#' 
#' @return A list containing target domain specific parameters.
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

    #### get Ximat - supports 0/1 and 1/2 encoding
    # Detect status encoding format
    if (all(primData$status %in% c(0, 1))) {
        # 0/1 encoding, event is 1
        event_indicator <- 1
    } else if (all(primData$status %in% c(1, 2))) {
        # 1/2 encoding, event is 2
        event_indicator <- 2
    } else {
        stop("Unsupported status encoding. Please use 0/1 or 1/2 encoding.")
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
