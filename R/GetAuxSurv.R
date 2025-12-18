#' Estimate Source Domain Parameters using Standard Cox or Sparse Cox Regression
#'
#' @description
#' Estimates source domain parameters. Automatically detects whether to use the sparse version
#' based on data dimensions, or forces a specific version.
#'
#' @param auxData Source domain data, containing time, status, and covariate columns.
#' @param weights Sample weights, optional.
#' @param cov Vector of covariate names.
#' @param use_sparse Logical. If NULL, detected automatically based on data dimensions.
#'
#' @return A list containing estR (regression coefficients) and q (baseline hazard parameters).
#'
#' @importFrom stats as.formula
#' @importFrom survival Surv coxph basehaz
#' @export
GetAuxSurv <- function(auxData, weights = NULL, cov = c("X1", "X2"), use_sparse = NULL) {

  # Automatically detect whether to use sparse version
  if (is.null(use_sparse)) {
    n_samples <- nrow(auxData)
    n_features <- length(cov)

    # Check event rate
    event_rate <- mean(auxData$status == 2) # Assuming 1/2 encoding
    if (all(auxData$status %in% c(0, 1))) {
      event_rate <- mean(auxData$status == 1)
    }

    # Use sparse version if features are many and event rate is sufficient
    # Threshold: more than 20 features and at least 10% events
    use_sparse <- (n_features > 20 && event_rate > 0.1)
  }

  if (use_sparse) {
    # Call the function defined in R/GetAuxSurv_Sparse.R
    return(GetAuxSurv_Sparse(auxData, cov = cov, weights = weights))
  } else {
    # Standard Cox Regression

    # Determine status target
    status_target <- if (all(auxData$status %in% c(0, 1))) 1 else 2

    formula_str <- paste("survival::Surv(time, status == ", status_target, ") ~ ", paste(cov, collapse = "+"), sep="")

    res.cox <- survival::coxph(stats::as.formula(formula_str), data = auxData, weights = weights)
    bhest <- survival::basehaz(res.cox, centered=FALSE) ## get baseline cumulative hazards
    estR <- res.cox$coefficients
    q <- data.frame(cumHazards = bhest$hazard,
                    breakPoints = bhest$time)
    return(list(estR = estR,
                q = q))
  }
}
