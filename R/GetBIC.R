#' Calculate Bayesian Information Criterion (BIC)
#' 
#' @description 
#' Calculates the BIC value for the TransCox model, used for model selection and parameter tuning.
#' 
#' @param status Vector of event status.
#' @param CovData Matrix of covariate data.
#' @param hazards Vector of hazard functions.
#' @param newBeta New regression coefficients.
#' @param newHaz New hazard functions.
#' @param eta Vector of eta parameters.
#' @param xi Vector of xi parameters.
#' @param cutoff Threshold for determining non-zero parameters.
#' @param lambda1 L1 penalty parameter for eta (used for adaptive thresholding).
#' @param lambda2 L1 penalty parameter for xi (used for adaptive thresholding).
#' @param lambda_beta L1 penalty parameter for beta_t (used for adaptive thresholding).
#' 
#' @return BIC value.
#' @export
GetBIC <- function(status, CovData, hazards,
                   newBeta,
                   newHaz,
                   eta,
                   xi,
                   cutoff = 1e-5,
                   lambda1 = NULL,
                   lambda2 = NULL,
                   lambda_beta = NULL) {
    
    # Performance optimization: pre-calculate common values
    n <- length(status)
    log_n <- log(n)
    
    # Fast calculation of log-likelihood
    Logl <- GetLogLike(status = status,
                       CovData = CovData,
                       hazards = hazards,
                       newBeta = newBeta,
                       newHaz = newHaz)
    
    # Check validity of log-likelihood
    if (is.na(Logl) || is.infinite(Logl)) {
        return(Inf)  # Return infinite BIC for invalid models
    }
    
    # Adaptive threshold optimization: using vectorized operations
    cutoff_eta <- if (!is.null(lambda1)) max(cutoff, 0.1 * lambda1) else cutoff
    cutoff_xi  <- if (!is.null(lambda2)) max(cutoff, 0.1 * lambda2) else cutoff
    cutoff_beta <- if (!is.null(lambda_beta)) max(cutoff, 0.1 * lambda_beta) else cutoff

    # Vectorized calculation of effective parameter counts
    K_eta <- sum(abs(eta) > cutoff_eta, na.rm = TRUE)
    K_xi  <- sum(abs(xi) > cutoff_xi, na.rm = TRUE)
    K_beta <- if (!is.null(newBeta)) sum(abs(newBeta) > cutoff_beta, na.rm = TRUE) else 0
    K_total <- K_eta + K_xi + K_beta
    
    # Numerical stability check
    if (K_total < 0 || K_total > n) {
        return(Inf)  # Abnormal parameter count
    }
    
    # Optimized BIC calculation: avoid re-calculating log(n)
    BIC <- K_total * log_n - 2 * Logl
    
    # Final numerical check
    if (is.na(BIC) || is.infinite(BIC)) {
        return(Inf)
    }
    
    return(BIC)
}
