#' Calculate Log-Likelihood Function
#' 
#' @description 
#' Calculates the log-likelihood value for the TransCox model, supporting different status encoding formats.
#' 
#' @param status Vector of event status, supports 0/1 or 1/2 encoding.
#' @param CovData Matrix of covariate data.
#' @param hazards Vector of original hazard functions.
#' @param newBeta Vector of new regression coefficients.
#' @param newHaz Vector of new hazard functions.
#' 
#' @return Log-likelihood value.
#' @export
GetLogLike <- function(status, CovData, hazards,
                       newBeta,
                       newHaz) {
    XB = as.matrix(CovData) %*% newBeta
    newCumH <- dQtocumQ(newHaz, status)
    
    # Prevent numerical issues with log function
    # Ensure values in newHaz are positive to avoid log(0) or log(negative)
    newHaz_safe <- pmax(newHaz, 1e-10)  # Set values less than 1e-10 to 1e-10
    
    # Detect status encoding format
    if (all(status %in% c(0, 1))) {
        # 0/1 encoding, event is 1
        event_indicator <- 1
    } else if (all(status %in% c(1, 2))) {
        # 1/2 encoding, event is 2
        event_indicator <- 2
    } else {
        stop("Unsupported status encoding. Please use 0/1 or 1/2 encoding.")
    }
    
    LogL <- sum(XB[status == event_indicator] + log(newHaz_safe)) - sum(newCumH * exp(XB))
    return(LogL)
}
