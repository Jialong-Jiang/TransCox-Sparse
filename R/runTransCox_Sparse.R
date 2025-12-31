#' Sparse TransCox Model for High-Dimensional Survival Analysis
#'
#' @description
#' This function implements the sparse TransCox model for high-dimensional survival analysis
#' with transfer learning. It integrates source domain (auxiliary) data to improve prediction
#' performance on the target domain (primary) data through regularized Cox regression with
#' L1 penalties.
#'
#' @param primData A data.frame containing the target domain survival data. Must include
#'    survival time, event status, and covariates.
#' @param auxData A data.frame containing the source domain survival data with the same
#'    structure as primData.
#' @param cov A character vector specifying the names of covariates to be used in the model.
#'    Default is c("X1", "X2").
#' @param statusvar A character string specifying the name of the event status variable.
#'    Default is "status".
#' @param lambda1 Numeric. L1 penalty parameter for eta (auxiliary parameter). If NULL,
#'    will be automatically selected via BIC. Default is NULL.
#' @param lambda2 Numeric. L1 penalty parameter for xi (transfer parameter). If NULL,
#'    will be automatically selected via BIC. Default is NULL.
#' @param lambda_beta Numeric. L1 penalty parameter for beta_t (target parameter). If NULL,
#'    will be automatically selected via BIC. Default is NULL.
#' @param learning_rate Numeric. Learning rate for the optimization algorithm. Default is 0.004.
#' @param nsteps Integer. Maximum number of optimization steps. Default is 200.
#' @param auto_tune Logical. Whether to automatically tune hyperparameters using BIC.
#'    Default is TRUE.
#' @param use_sparse Logical. Deprecated parameter, kept for compatibility. This function
#'    ALWAYS runs in sparse mode. Default is TRUE.
#' @param verbose Logical. Whether to display detailed progress information. Default is TRUE.
#' @param tolerance Numeric. Convergence tolerance for the optimization algorithm.
#'    Default is 1e-6.
#' @param early_stopping Logical. Whether to enable early stopping mechanism. Default is TRUE.
#' @param adaptive_lr Logical. Whether to use adaptive learning rate. Default is TRUE.
#' @param parallel Logical. Whether to use parallel computation for parameter search. Default is FALSE.
#' @param n_cores Integer. Number of cores for parallel computation. If NULL, detected automatically.
#' @param threshold_c Numeric. Constant for theoretical hard thresholding (tau = C * sqrt(log(p)/n)).
#'    Default is 0.5. Controls the post-hoc sparsity level.
#'
#' @return A list containing the estimated coefficients and convergence information.
#'
#' @importFrom reticulate source_python
#' @export
runTransCox_Sparse <- function(primData, auxData,
                               cov = c("X1", "X2"),
                               statusvar = "status",
                               lambda1 = NULL,
                               lambda2 = NULL,
                               lambda_beta = NULL,
                               learning_rate = 0.001,
                               nsteps = 500,
                               auto_tune = TRUE,
                               use_sparse = TRUE, # Parameter kept for interface compatibility, but ignored logic-wise
                               verbose = TRUE,
                               tolerance = 1e-6,
                               early_stopping = TRUE,
                               adaptive_lr = TRUE,
                               parallel = FALSE,
                               n_cores = NULL,
                               threshold_c = 0.5) {

  # --- 1. Enforce Sparse Mode & Load Python ---
  # We strictly enforce sparse mode regardless of data dimensions or use_sparse flag.

  if (!exists("TransCox_Sparse", mode = "function")) {
    py_path <- system.file("python", "TransCoxFunction_Sparse.py", package = "TransCoxSparse")
    # Fallback for dev mode
    if (py_path == "") py_path <- file.path(getwd(), "inst", "python", "TransCoxFunction_Sparse.py")

    if (file.exists(py_path)) {
      reticulate::source_python(py_path, envir = environment())
    } else {
      stop("Python script TransCoxFunction_Sparse.py not found. Please ensure the package is installed correctly.")
    }
  }

  # --- 2. Load Helper R Functions (if not in package namespace) ---
  # In a proper package structure, these should be imported via NAMESPACE.
  # These checks are for development/source usage.
  if (!exists("GetPrimaryParam") && file.exists(file.path(getwd(), "R", "GetPrimaryParam.R"))) {
    source(file.path(getwd(), "R", "GetPrimaryParam.R"))
  }
  if (!exists("GetAuxSurv_Sparse") && file.exists(file.path(getwd(), "R", "GetAuxSurv.R"))) {
    source(file.path(getwd(), "R", "GetAuxSurv.R"))
  }
  if (!exists("SelParam_By_BIC_Sparse") && file.exists(file.path(getwd(), "R", "SelParam_By_BIC_Sparse.R"))) {
    source(file.path(getwd(), "R", "SelParam_By_BIC_Sparse.R"))
  }

  # --- 3. Parameter Tuning (BIC) ---
  need_tune <- auto_tune && (
    is.null(lambda1) || is.null(lambda2) || is.null(lambda_beta) ||
      length(lambda1) > 1 || length(lambda2) > 1 || length(lambda_beta) > 1
  )

  if (need_tune) {
    if (verbose) cat("Starting automatic parameter tuning (BIC)...\n")

    # Defaults ranges if not specified
    l1_range <- if(is.null(lambda1)) c(0.1, 0.5, 1.0, 2.0) else lambda1
    l2_range <- if(is.null(lambda2)) c(0.1, 0.5, 1.0, 2.0) else lambda2
    lb_range <- if(is.null(lambda_beta)) c(0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2) else lambda_beta

    # STRICTLY CALL SPARSE BIC FUNCTION
    bic_result <- SelParam_By_BIC_Sparse(
      primData = primData,
      auxData = auxData,
      cov = cov,
      statusvar = statusvar,
      lambda1_vec = l1_range,
      lambda2_vec = l2_range,
      lambda_beta_vec = lb_range,
      learning_rate = learning_rate,
      nsteps = nsteps,
      verbose = verbose,
      parallel = parallel,
      n_cores = n_cores,
      threshold_c = threshold_c
    )

    lambda1 <- bic_result$best_lambda1
    lambda2 <- bic_result$best_lambda2
    lambda_beta <- bic_result$best_lambda_beta

    # If BIC result contains final model, return directly to save computation
    if (!is.null(bic_result$final_beta)) {
      final_beta_vec <- bic_result$final_beta
      nonzero_cnt <- sum(abs(final_beta_vec) > 1e-8)

      result <- list(
        eta = bic_result$final_eta,
        xi = bic_result$final_xi,
        new_beta = final_beta_vec,
        new_IntH = if (!is.null(bic_result$final_xi)) bic_result$final_xi else rep(0, sum(auxData[[statusvar]] == 2)),
        source_estR = if (!is.null(bic_result$source_estR)) bic_result$source_estR else rep(0, length(cov)),
        lambda1_used = lambda1,
        lambda2_used = lambda2,
        lambda_beta_used = lambda_beta,
        convergence_info = bic_result$convergence_info,
        bic_result = bic_result,
        nonzero_count = nonzero_cnt,
        sparsity_ratio = 1 - nonzero_cnt / length(final_beta_vec),
        use_sparse = TRUE
      )
      class(result) <- "TransCox_Sparse"
      return(result)
    }
  }

  # --- 4. Parameter Defaults (if not tuned) ---
  if (is.null(lambda1)) lambda1 <- 0.01
  if (is.null(lambda2)) lambda2 <- 0.01
  if (is.null(lambda_beta)) {
    # Default logic for lambda_beta in sparse mode
    n_features <- length(cov)
    n_samples <- nrow(primData)
    if (n_features > n_samples / 2) {
      lambda_beta <- 0.03
    } else {
      lambda_beta <- 0.02
    }
  }

  # Learning rate adjustments
  if (adaptive_lr && nsteps > 100) {
    learning_rate <- learning_rate * 0.8
    if (verbose) cat("Adaptive learning rate adjusted to:", learning_rate, "\n")
  }

  if (early_stopping && nsteps > 500) {
    nsteps <- min(nsteps, 300)
    if (verbose) cat("Early stopping enabled, max steps adjusted to:", nsteps, "\n")
  }

  # --- 5. Data Preparation (Sparse Path Only) ---

  # Estimate Source Domain Parameters (Sparse)
  Cout <- GetAuxSurv_Sparse(auxData, cov = cov)

  # Calculate Target Domain Parameters
  Pout <- GetPrimaryParam(primData, q = Cout$q, estR = Cout$estR)

  # Prepare Data
  CovData <- Pout$primData[, cov]
  status <- Pout$primData[, statusvar]
  cumH <- Pout$primData$fullCumQ
  hazards <- Pout$dQ$dQ

  if (verbose) cat("Preprocessing data for optimized transfer...\n")

  # Preprocess data matrix (using data.matrix to handle Factors safely)
  CovData_optimized <- data.matrix(CovData)
  storage.mode(CovData_optimized) <- "double"

  # Preprocess vector data
  cumH_optimized <- as.double(cumH)
  hazards_optimized <- as.double(hazards)
  status_optimized <- as.integer(status)
  estR_optimized <- as.double(Pout$estR)

  Xinn_optimized <- data.matrix(Pout$Xinn)
  storage.mode(Xinn_optimized) <- "double"

  # --- 6. Call Python Sparse Function ---

  params_package <- list(
    CovData = CovData_optimized,
    cumH = cumH_optimized,
    hazards = hazards_optimized,
    status = status_optimized,
    estR = estR_optimized,
    Xinn = Xinn_optimized,
    lambda1 = as.double(lambda1),
    lambda2 = as.double(lambda2),
    lambda_beta = as.double(lambda_beta),
    learning_rate = as.double(learning_rate),
    nsteps = as.integer(nsteps),
    tolerance = as.double(tolerance),
    verbose = verbose,
    threshold_c = as.double(threshold_c)
  )

  if (verbose) cat("Calling optimized sparse TransCox function...\n")

  # Ensure Python function is loaded (Double check)
  if (!exists("TransCox_Sparse", mode = "function")) {
    py_path <- system.file("python", "TransCoxFunction_Sparse.py", package = "TransCoxSparse")
    if(py_path != "") reticulate::source_python(py_path, envir = environment())
  }

  trans_result <- do.call(TransCox_Sparse, params_package)

  eta <- trans_result[[1]]
  xi <- trans_result[[2]]
  new_beta <- trans_result[[3]]
  convergence_info <- trans_result[[4]]

  # --- 7. Result Packaging ---
  nonzero_beta <- sum(abs(new_beta) > 1e-8, na.rm = TRUE)
  sparsity_ratio <- 1 - nonzero_beta / length(new_beta)

  sparsity_warnings <- character(0)
  if (!is.na(nonzero_beta) && nonzero_beta == 0) {
    sparsity_warnings <- c(sparsity_warnings, "Warning: All coefficients zeroed out")
    if (verbose) cat("Warning: All coefficients zeroed out.\n")
  }

  result <- list(
    eta = eta,
    xi = xi,
    new_beta = new_beta,
    new_IntH = Pout$dQ$dQ + xi,
    time = Pout$primData[status == 2, "time"],
    source_estR = Pout$estR,
    lambda1_used = lambda1,
    lambda2_used = lambda2,
    lambda_beta_used = lambda_beta,
    nonzero_count = nonzero_beta,
    sparsity_ratio = sparsity_ratio,
    convergence_info = convergence_info,
    sparsity_warnings = sparsity_warnings,
    use_sparse = TRUE
  )

  class(result) <- "TransCox_Sparse"
  return(result)
}
