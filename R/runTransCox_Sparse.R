#' Sparse TransCox Model for High-Dimensional Survival Analysis (PGD Version)
#'
#' @description
#' This function implements the sparse TransCox model using Proximal Gradient Descent (PGD).
#' It handles high-dimensional transfer learning with an ADDITIVE hazard model (JASA consistent).
#'
#' @param primData Target domain data.frame (time, status, covariates).
#' @param auxData Source domain data.frame.
#' @param cov Character vector of covariate names.
#' @param statusvar Name of status variable.
#' @param lambda1 Numeric. Penalty for Transfer (beta - beta_source). Scaled by 1/N_events.
#' @param lambda2 Numeric. Penalty for Xi (Baseline Hazard deviation). Scaled by 1/N_events.
#' @param lambda_beta Numeric. Penalty for Global Sparsity (beta). Scaled by 1/N_events.
#' @param learning_rate Numeric. Step size for PGD. Default 0.01.
#' @param nsteps Integer. Optimization steps. Default 2000.
#' @param auto_tune Logical. Whether to use BIC.
#' @param verbose Logical. Print progress.
#' @param tolerance Numeric. Convergence tolerance.
#' @param use_sparse Logical. Whether to use sparse implementation.
#' @param parallel Logical. Whether to run parameter tuning in parallel.
#' @param n_cores Integer. Number of cores for parallel processing.
#'
#' @return List containing estimated coefficients and model details.
#' @export
runTransCox_Sparse <- function(primData, auxData,
                               cov = c("X1", "X2"),
                               statusvar = "status",
                               lambda1 = NULL,
                               lambda2 = NULL,
                               lambda_beta = NULL,
                               learning_rate = 0.01,
                               nsteps = 2000,
                               auto_tune = TRUE,
                               use_sparse = TRUE,
                               verbose = TRUE,
                               tolerance = 1e-7,
                               parallel = FALSE,
                               n_cores = NULL) {

  # --- 1. Python Environment Check ---
  if (!exists("TransCox_Sparse", mode = "function")) {
    py_path <- system.file("python", "TransCoxFunction_Sparse.py", package = "TransCoxSparse")
    if (py_path == "") py_path <- file.path(getwd(), "inst", "python", "TransCoxFunction_Sparse.py")

    if (file.exists(py_path)) {
      reticulate::source_python(py_path, envir = environment())
    } else {
      stop("Python script TransCoxFunction_Sparse.py not found.")
    }
  }

  # --- 2. Load Helpers (Dev Mode) ---
  if (!exists("GetPrimaryParam") && file.exists(file.path(getwd(), "R", "GetPrimaryParam.R")))
    source(file.path(getwd(), "R", "GetPrimaryParam.R"))
  if (!exists("GetAuxSurv_Sparse") && file.exists(file.path(getwd(), "R", "GetAuxSurv.R")))
    source(file.path(getwd(), "R", "GetAuxSurv.R"))
  if (!exists("SelParam_By_BIC_Sparse") && file.exists(file.path(getwd(), "R", "SelParam_By_BIC_Sparse.R")))
    source(file.path(getwd(), "R", "SelParam_By_BIC_Sparse.R"))

  # --- 3. Parameter Tuning (BIC) ---
  need_tune <- auto_tune && (
    is.null(lambda1) || is.null(lambda2) || is.null(lambda_beta) ||
      length(lambda1) > 1 || length(lambda2) > 1 || length(lambda_beta) > 1
  )

  if (need_tune) {
    if (verbose) cat("Starting automatic parameter tuning (BIC)...\n")

    # Ranges adjusted for Normalized Loss
    l1_range <- if(is.null(lambda1)) c(0.01, 0.05, 0.1, 0.2, 0.5) else lambda1
    l2_range <- if(is.null(lambda2)) c(0.01, 0.05, 0.1) else lambda2
    lb_range <- if(is.null(lambda_beta)) c(0.005, 0.01, 0.02, 0.05, 0.1) else lambda_beta

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
      n_cores = n_cores
    )

    lambda1 <- bic_result$best_lambda1
    lambda2 <- bic_result$best_lambda2
    lambda_beta <- bic_result$best_lambda_beta

    # Return directly if BIC computed the final model
    if (!is.null(bic_result$final_beta)) {
      final_beta_vec <- bic_result$final_beta
      nonzero_cnt <- sum(abs(final_beta_vec) > 1e-8)

      base_haz <- if(!is.null(bic_result$base_haz)) bic_result$base_haz else rep(1e-5, length(bic_result$final_xi))

      # [SCIENTIFIC FIX] Additive Model: h_new = h_old + xi
      new_int_h <- base_haz + bic_result$final_xi

      result <- list(
        eta = bic_result$final_eta,
        xi = bic_result$final_xi,
        new_beta = final_beta_vec,
        new_IntH = new_int_h, # Fixed
        source_estR = bic_result$source_estR,
        lambda1_used = lambda1,
        lambda2_used = lambda2,
        lambda_beta_used = lambda_beta,
        convergence_info = bic_result$convergence_info,
        nonzero_count = nonzero_cnt,
        sparsity_ratio = 1 - nonzero_cnt / length(final_beta_vec)
      )
      class(result) <- "TransCox_Sparse"
      return(result)
    }
  }

  # --- 4. Defaults ---
  if (is.null(lambda1)) lambda1 <- 0.05
  if (is.null(lambda2)) lambda2 <- 0.01
  if (is.null(lambda_beta)) lambda_beta <- 0.02

  # --- 5. Data Preparation ---
  Cout <- GetAuxSurv_Sparse(auxData, cov = cov)
  Pout <- GetPrimaryParam(primData, q = Cout$q, estR = Cout$estR)

  CovData <- data.matrix(Pout$primData[, cov])
  status <- as.integer(Pout$primData[, statusvar])
  cumH <- as.double(Pout$primData$fullCumQ)
  hazards <- as.double(Pout$dQ$dQ)
  estR <- as.double(Pout$estR)
  Xinn <- data.matrix(Pout$Xinn)

  # --- 6. Python Execution ---
  params_package <- list(
    CovData = CovData,
    cumH = cumH,
    hazards = hazards,
    status = status,
    estR = estR,
    Xinn = Xinn,
    lambda1 = as.double(lambda1),
    lambda2 = as.double(lambda2),
    lambda_beta = as.double(lambda_beta),
    learning_rate = as.double(learning_rate),
    nsteps = as.integer(nsteps),
    tolerance = as.double(tolerance),
    verbose = verbose
  )

  if (verbose) cat("Calling Proximal Gradient TransCox (Sparse)...\n")

  if (!exists("TransCox_Sparse", mode = "function")) {
    reticulate::source_python(system.file("python", "TransCoxFunction_Sparse.py", package = "TransCoxSparse"))
  }

  trans_result <- do.call(TransCox_Sparse, params_package)

  eta <- trans_result[[1]]
  xi <- trans_result[[2]]
  new_beta <- trans_result[[3]]
  convergence_info <- trans_result[[4]]

  # --- 7. Result Packaging ---
  nonzero_beta <- sum(abs(new_beta) > 1e-8, na.rm = TRUE)

  # [SCIENTIFIC FIX] Additive Model: h_new = h_old + xi
  new_IntH <- hazards + xi

  result <- list(
    eta = eta,
    xi = xi,
    new_beta = new_beta,
    new_IntH = new_IntH, # Fixed
    time = Pout$primData[status == 2, "time"],
    source_estR = estR,
    lambda1_used = lambda1,
    lambda2_used = lambda2,
    lambda_beta_used = lambda_beta,
    nonzero_count = nonzero_beta,
    sparsity_ratio = 1 - nonzero_beta / length(new_beta),
    convergence_info = convergence_info
  )

  class(result) <- "TransCox_Sparse"
  return(result)
}
