#' Calculate Confidence Interval Coverage
#' 
#' @description 
#' Checks if the 95% confidence interval of the parameter estimate covers the true value.
#' 
#' @param estimate Parameter estimate.
#' @param trueBeta True parameter value.
#' @param SE Standard error.
#' 
#' @return 0 or 1, indicating whether the true value is covered.
#' @export
GetCoverage <- function(estimate, trueBeta,
                        SE) {
    LL <- estimate - 1.96*SE
    UU <- estimate + 1.96*SE
    return(as.numeric(trueBeta<UU & trueBeta>LL))
}
