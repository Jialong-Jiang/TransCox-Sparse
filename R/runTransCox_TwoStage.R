#' Two-Stage TransCox Algorithm with Post-Selection Refitting
#'
#' @description
#' Implements a "Relaxed Lasso" or "Post-Lasso" strategy for high-dimensional survival analysis
#' with transfer learning.
#'
#' The process consists of two main stages:
#' \enumerate{
#'   \item Screening (Stage 1): Uses the TransCox algorithm (penalized transfer learning) with a
#'   high-recall configuration to select a candidate set of active variables.
#'   \item Refitting (Stage 2): Performs an unpenalized standard Cox Proportional Hazards regression
#'   (CoxPH) on the selected subset. This step removes shrinkage bias and filters out false positives
#'   based on statistical significance (P-value).
#' }
#'
#' @param primData A data.frame containing the target domain data (primary data). Must contain time, status, and covariates.
#' @param auxData A data.frame containing the source domain data (auxiliary data).
#' @param cov A character vector of feature names to be included in the analysis.
#' @param statusvar A character string specifying the name of the status variable (event indicator).
#' @param p_value_threshold A numeric value specifying the P-value threshold for filtering variables
#'        in the second stage (default is 0.05). Variables with P-values above this threshold are discarded.
#' @param lambda1 Numeric. Penalty for Transfer (beta - beta_source). Scaled by 1/N_events.
#' @param lambda2 Numeric. Penalty for Xi (Baseline Hazard deviation). Scaled by 1/N_events.
#' @param lambda_beta Numeric. Penalty for Global Sparsity (beta). Scaled by 1/N_events.
#' @param learning_rate Numeric. Step size for Proximal Gradient Descent. Default is 0.001.
#' @param nsteps Integer. Maximum number of optimization steps. Default is 5000.
#' @param tolerance Numeric. Convergence tolerance for the PGD algorithm. Default is 1e-7.
#' @param parallel Logical. Whether to run internal calculations in parallel (if applicable). Default is FALSE.
#' @param n_cores Integer. Number of cores for parallel processing. Default is NULL.
#' @param verbose Logical. If TRUE, prints progress messages to the console during execution.
#' @param auto_tune Logical. Whether to use BIC.
#'
#' @return A list containing the following components:
#' \item{new_beta}{The estimated coefficient vector (numeric vector of length P). These are unbiased estimates from the final refitted model.}
#' \item{new_IntH}{A data.frame containing the re-estimated baseline cumulative hazard with columns \code{time} and \code{cumQ_upd}.}
#' \item{stage1_vars}{A character vector of variables selected in the first screening stage (Stage 1).}
#' \item{final_vars}{A character vector of variables remaining after P-value filtering (Stage 2).}
#' \item{fit_obj}{The final \code{coxph} object from the second stage, useful for standard error and summary extraction.}
#'
#' @importFrom survival coxph basehaz
#' @export
runTransCox_TwoStage <- function(primData, auxData, cov, statusvar,
                                 p_value_threshold = 0.05,
                                 lambda1 = NULL,
                                 lambda2 = NULL,
                                 lambda_beta = NULL,
                                 learning_rate = 0.001,
                                 nsteps = 5000,
                                 tolerance = 1e-7,
                                 parallel = FALSE,
                                 n_cores = NULL,
                                 verbose = FALSE,
                                 auto_tune = TRUE) {

  # --- Stage 1: Screening (Variable Selection) ---
  if(verbose) cat(">> [Stage 1] Running TransCox Lasso...\n")

  # Force auto_tune = FALSE to strictly control parameters
  # Force use_sparse = TRUE for variable selection
  res_stage1 <- runTransCox_Sparse(
    primData = primData,
    auxData = auxData,
    cov = cov,
    statusvar = statusvar,
    lambda1 = lambda1,
    lambda2 = lambda2,
    lambda_beta = lambda_beta,
    learning_rate = learning_rate,
    nsteps = nsteps,
    tolerance = tolerance,
    parallel = parallel,
    n_cores = n_cores,
    auto_tune = auto_tune,
    use_sparse = TRUE,
    verbose = verbose
  )

  # Extract Stage 1 coefficients
  beta_s1 <- as.vector(res_stage1$new_beta)

  # Use numerical tolerance for truncation instead of heuristic cutoff
  selected_idx <- which(abs(beta_s1) > 1e-6)
  selected_vars <- cov[selected_idx]

  n_selected <- length(selected_vars)
  n_samples <- nrow(primData)

  # --- Failsafe: Dimensionality Check ---
  # Assessment: If selected variables > sample size, CoxPH cannot run.
  # Strategy: Apply Top-K truncation to ensure model identifiability.
  # Using N/2 as a conservative safety limit to preserve degrees of freedom.
  safe_limit <- floor(n_samples / 2)

  if (n_selected == 0) {
    if(verbose) cat("!! [Warning] No variables selected in Stage 1. Returning NULL model.\n")
    return(res_stage1) # Cannot refit, return original result
  }

  if (n_selected > safe_limit) {
    if(verbose) cat(sprintf("!! [Warning] Selected %d vars, exceeding safety limit (%d). Applying Top-K truncation.\n", n_selected, safe_limit))
    # Sort by absolute magnitude and keep the top safe_limit variables
    ord <- order(abs(beta_s1[selected_idx]), decreasing = TRUE)
    selected_vars <- selected_vars[ord[1:safe_limit]]
  }

  # --- Stage 2: Unpenalized Refitting (Intermediate) ---
  if(verbose) cat(sprintf(">> [Stage 2] Refitting CoxPH with %d variables...\n", length(selected_vars)))

  # Construct refitting dataset
  refit_df <- cbind(primData[, c("time", statusvar)],
                    as.matrix(primData[, selected_vars, drop=FALSE]))

  # Run standard CoxPH
  fmla <- as.formula(paste("Surv(time, ", statusvar, ") ~ ."))
  fit_s2 <- tryCatch(coxph(fmla, data = refit_df, x = TRUE, y = TRUE), error = function(e) NULL)

  if(is.null(fit_s2)) {
    if(verbose) cat("!! [Error] CoxPH refit failed (singular matrix). Returning Stage 1 results.\n")
    return(res_stage1)
  }

  # --- Stage 3: P-value Filtering ---
  summ <- summary(fit_s2)

  # Extract P-values (Handle potential backticks in variable names)
  coef_mat <- summ$coefficients
  pvals <- coef_mat[, "Pr(>|z|)"]
  raw_names <- rownames(coef_mat)
  clean_names <- gsub("`", "", raw_names)

  # Filter variables where P < threshold
  final_vars <- clean_names[pvals < p_value_threshold]

  if(verbose) cat(sprintf(">> [Stage 3] P-value filtering: %d -> %d variables.\n", length(selected_vars), length(final_vars)))

  # If no variables remain after filtering, return an empty model structure
  if(length(final_vars) == 0) {
    res_empty <- res_stage1
    res_empty$new_beta <- rep(0, length(cov)) # All zeros
    return(res_empty)
  }

  # --- Stage 4: Final Estimation & Reconstruction ---
  # Perform final Cox fit on the cleaned variable set to get pure coefficients and baseline hazard
  final_df <- cbind(primData[, c("time", statusvar)],
                    as.matrix(primData[, final_vars, drop=FALSE]))
  fit_final <- coxph(as.formula(paste("Surv(time, ", statusvar, ") ~ .")),
                     data = final_df, x = TRUE, y = TRUE)

  # 1. Reconstruct Beta (Map back to P-dimensional vector)
  final_beta_vec <- rep(0, length(cov))
  final_coefs <- coef(fit_final)
  final_names <- gsub("`", "", names(final_coefs))

  for(k in 1:length(final_names)) {
    var_name <- final_names[k]
    idx <- which(cov == var_name)
    if(length(idx) > 0) {
      final_beta_vec[idx] <- final_coefs[k]
    }
  }

  # 2. Re-calculate Baseline Hazard (new_IntH)
  # TransCox output format requires data.frame(time, cumQ_upd)
  # Use Breslow estimator via basehaz (centered=FALSE matches raw coefficients)
  bh <- basehaz(fit_final, centered = FALSE)
  new_IntH <- data.frame(
    time = bh$time,
    cumQ_upd = bh$hazard
  )

  # --- Construct Final Output Object ---
  # Maintain structural compatibility with runTransCox_Sparse
  res_final <- list(
    new_beta = final_beta_vec,   # New unbiased coefficients
    new_IntH = new_IntH,         # New baseline hazard

    # Parameters from PGD optimization are invalid after refitting,eta = NULL,xi = NULL
    # Additional info for diagnostics
    eta = NULL,
    xi = NULL,
    stage1_vars = selected_vars,
    final_vars = final_vars,
    fit_obj = fit_final          # Return coxph object for summary() access
  )

  return(res_final)
}
