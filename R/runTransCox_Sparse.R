#' Sparse TransCox Model for High-Dimensional Survival Analysis
#'
#' @description
#' This function implements the sparse TransCox model for high-dimensional survival analysis
#' with transfer learning. It integrates source domain (auxiliary) data to improve prediction
#' performance on the target domain (primary) data through regularized Cox regression with
#' L1 penalties.
#'
#' @details
#' The sparse TransCox model addresses the challenge of limited sample sizes in survival
#' analysis by leveraging information from related source domains. The model uses L1
#' regularization to achieve sparsity and automatic feature selection, making it suitable
#' for high-dimensional data where the number of features exceeds the number of samples.
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
#' @param use_sparse Logical. Whether to force the use of sparse implementation. If NULL,
#'    automatically determined based on data dimensions. Default is NULL.
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
#' @return A list containing the following components:
#' \describe{
#'    \item{beta_t}{Estimated coefficients for the target domain}
#'    \item{eta}{Estimated auxiliary parameters}
#'    \item{xi}{Estimated transfer parameters}
#'    \item{lambda1}{Used L1 penalty for eta}
#'    \item{lambda2}{Used L1 penalty for xi}
#'    \item{lambda_beta}{Used L1 penalty for beta_t}
#'    \item{convergence}{Convergence information}
#'    \item{sparse_info}{Information about sparsity patterns}
#' }
#'
#' @examples
#' \dontrun{
#' # Generate example data
#' data <- generate_sparse_survival_data(n_main = 100, n_aux = 200, p = 50)
#'
#' # Fit sparse TransCox model
#' result <- runTransCox_Sparse(
#'    primData = data$prim_data,
#'    auxData = data$aux_data,
#'    cov = paste0("X", 1:50),
#'    statusvar = "status"
#' )
#'
#' # View results
#' print(result)
#' }
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
                               use_sparse = NULL,
                               verbose = TRUE,
                               tolerance = 1e-6,
                               early_stopping = TRUE,
                               adaptive_lr = TRUE,
                               parallel = FALSE,
                               n_cores = NULL,
                               threshold_c = 0.5) {

  # Automatically detect whether to use sparse version
  if (is.null(use_sparse)) {
    n_samples <- nrow(primData)
    n_features <- length(cov)
    use_sparse <- (n_features > n_samples / 2)
  }

  # Load Python Functions
  if (use_sparse) {
    if (!exists("TransCox_Sparse", mode = "function")) {
      py_path <- system.file("python", "TransCoxFunction_Sparse.py", package = "TransCoxSparse")
      # Fallback for dev mode
      if (py_path == "") py_path <- file.path(getwd(), "inst", "python", "TransCoxFunction_Sparse.py")

      if (file.exists(py_path)) {
        reticulate::source_python(py_path, envir = environment())
      } else {
        warning("Python script TransCoxFunction_Sparse.py not found.")
      }
    }
  } else {
    if (!exists("TransCox", mode = "function")) {
      py_path <- system.file("python", "TransCoxFunction.py", package = "TransCoxSparse")
      # Fallback for dev mode
      if (py_path == "") py_path <- file.path(getwd(), "inst", "python", "TransCoxFunction.py")

      if (file.exists(py_path)) {
        reticulate::source_python(py_path, envir = environment())
      } else {
        warning("Python script TransCoxFunction.py not found.")
      }
    }
  }

  # Parameter Tuning
  need_tune <- auto_tune && (
    is.null(lambda1) || is.null(lambda2) || is.null(lambda_beta) ||
      length(lambda1) > 1 || length(lambda2) > 1 || length(lambda_beta) > 1
  )

  if (need_tune) {
    # Automatic Parameter Tuning
    if (use_sparse) {
      # Use sparse version BIC selection
      l1_range <- if(is.null(lambda1)) c(0.1, 0.5, 1.0, 2.0) else lambda1
      l2_range <- if(is.null(lambda2)) c(0.1, 0.5, 1.0, 2.0) else lambda2
      lb_range <- if(is.null(lambda_beta)) c(0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2) else lambda_beta

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

      # If BIC result contains final model, return directly
      if (!is.null(bic_result$final_beta)) {

        final_beta_vec <- bic_result$final_beta
        nonzero_cnt <- sum(abs(final_beta_vec) > 1e-8)

        result <- list(
          eta = bic_result$final_eta,
          xi = bic_result$final_xi,
          new_beta = final_beta_vec,
          new_IntH = if (!is.null(bic_result$final_xi)) bic_result$final_xi else rep(0, sum(auxData$status == 2)),
          source_estR = if (!is.null(bic_result$source_estR)) bic_result$source_estR else rep(0, length(cov)),
          lambda1_used = lambda1,
          lambda2_used = lambda2,
          lambda_beta_used = lambda_beta,
          convergence_info = bic_result$convergence_info,
          bic_result = bic_result,
          nonzero_count = nonzero_cnt,
          sparsity_ratio = 1 - nonzero_cnt / length(final_beta_vec),
          use_sparse = use_sparse
        )

        class(result) <- "TransCox_Sparse"
        return(result)
      }

    } else {
      # Use original version BIC selection
      l1_range <- if(is.null(lambda1)) c(0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1) else lambda1
      l2_range <- if(is.null(lambda2)) c(0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1) else lambda2

      bic_result <- SelParam_By_BIC(
        primData = primData,
        auxData = auxData,
        cov = cov,
        statusvar = statusvar,
        lambda1_vec = l1_range,
        lambda2_vec = l2_range,
        learning_rate = learning_rate,
        nsteps = nsteps
      )

      lambda1 <- bic_result$best_la1
      lambda2 <- bic_result$best_la2
      lambda_beta <- 0  # Original version does not use lambda_beta
    }
  }

  # Set default values if not tuned
  if (is.null(lambda1)) lambda1 <- 0.01
  if (is.null(lambda2)) lambda2 <- 0.01
  if (is.null(lambda_beta)) {
    # Adaptively set lambda_beta default based on data dimensions
    n_features <- length(cov)
    n_samples <- nrow(primData)
    if (use_sparse && n_features > n_samples / 2) {
      lambda_beta <- 0.03
    } else if (use_sparse) {
      lambda_beta <- 0.02
    } else {
      lambda_beta <- 0
    }
  }

  # Adaptive learning rate adjustment
  if (adaptive_lr && nsteps > 100) {
    learning_rate <- learning_rate * 0.8
    if (verbose) cat("Adaptive learning rate adjusted to:", learning_rate, "\n")
  }

  # Early stopping mechanism
  if (early_stopping && nsteps > 500) {
    nsteps <- min(nsteps, 300)
    if (verbose) cat("Early stopping enabled, max steps adjusted to:", nsteps, "\n")
  }

  # Final Model Fitting

  # Estimate Source Domain Parameters
  if (use_sparse) {
    Cout <- GetAuxSurv_Sparse(auxData, cov = cov)
  } else {
    Cout <- GetAuxSurv(auxData, cov = cov)
  }

  # Calculate Target Domain Parameters
  Pout <- GetPrimaryParam(primData, q = Cout$q, estR = Cout$estR)

  # Prepare Data
  CovData <- Pout$primData[, cov]
  status <- Pout$primData[, statusvar]
  cumH <- Pout$primData$fullCumQ
  hazards <- Pout$dQ$dQ

  # Optimized R-Python Interface Call
  if (verbose) cat("Preprocessing data for optimized transfer...\n")

  # Preprocess data matrix
  CovData_optimized <- as.matrix(CovData)
  storage.mode(CovData_optimized) <- "double"

  # Preprocess vector data
  cumH_optimized <- as.double(cumH)
  hazards_optimized <- as.double(hazards)
  status_optimized <- as.integer(status)
  estR_optimized <- as.double(Pout$estR)
  Xinn_optimized <- as.matrix(Pout$Xinn)
  storage.mode(Xinn_optimized) <- "double"

  # Check if using sparse version (handling vector parameters)
  use_sparse_version <- use_sparse && (
    (length(lambda_beta) > 1) ||
      (length(lambda_beta) == 1 && lambda_beta > 0)
  )

  if (use_sparse_version) {
    # Create optimized parameter package
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
      threshold_c = as.double(threshold_c) # Pass new parameter to Python
    )

    # Use sparse version
    if (verbose) cat("Calling optimized sparse TransCox function...\n")

    # Ensure TransCox_Sparse is available in current scope
    if (!exists("TransCox_Sparse", mode = "function")) {
      py_path <- system.file("python", "TransCoxFunction_Sparse.py", package = "TransCoxSparse")
      if (py_path == "") py_path <- file.path(getwd(), "inst", "python", "TransCoxFunction_Sparse.py")
      reticulate::source_python(py_path, envir = environment())
    }

    trans_result <- do.call(TransCox_Sparse, params_package)

    eta <- trans_result[[1]]
    xi <- trans_result[[2]]
    new_beta <- trans_result[[3]]
    convergence_info <- trans_result[[4]]

  } else {
    # Create optimized parameter package (original version)
    params_package_orig <- list(
      CovData = CovData_optimized,
      cumH = cumH_optimized,
      hazards = hazards_optimized,
      status = status_optimized,
      estR = estR_optimized,
      Xinn = Xinn_optimized,
      lambda1 = as.double(lambda1),
      lambda2 = as.double(lambda2),
      learning_rate = as.double(learning_rate),
      nsteps = as.integer(nsteps)
    )

    # Use original version
    if (verbose) cat("Calling optimized original TransCox function...\n")

    if (!exists("TransCox", mode = "function")) {
      py_path <- system.file("python", "TransCoxFunction.py", package = "TransCoxSparse")
      if (py_path == "") py_path <- file.path(getwd(), "inst", "python", "TransCoxFunction.py")
      reticulate::source_python(py_path, envir = environment())
    }

    trans_result <- do.call(TransCox, params_package_orig)

    eta <- trans_result[[1]]
    xi <- trans_result[[2]]
    new_beta <- estR_optimized + eta
    convergence_info <- NULL
  }

  # Calculate sparsity statistics
  nonzero_beta <- sum(abs(new_beta) > 1e-8, na.rm = TRUE)
  sparsity_ratio <- 1 - nonzero_beta / length(new_beta)

  # Sparsity checks
  sparsity_warnings <- character(0)
  if (!is.na(nonzero_beta) && nonzero_beta == 0) {
    sparsity_warnings <- c(sparsity_warnings, "Warning: All coefficients zeroed out")
    if (verbose) cat("Warning: All coefficients zeroed out, suggest lowering lambda_beta\n")
  }

  # Return Results
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
    use_sparse = use_sparse
  )

  class(result) <- "TransCox_Sparse"
  return(result)
}

#' Print TransCox_Sparse Results
#'
#' @description
#' Print method for TransCox_Sparse object.
#'
#' @param x A TransCox_Sparse object.
#' @param ... Additional arguments passed to print.
#' @export
print.TransCox_Sparse <- function(x, ...) {
  cat("TransCox Sparse Results\n")
  cat("Features:", length(x$new_beta), "\n")
  cat("Non-zero coefficients:", x$nonzero_count, "\n")
  cat("Sparsity:", round(x$sparsity_ratio * 100, 2), "%\n")
  cat("Parameters: lambda1=", x$lambda1_used, ", lambda2=", x$lambda2_used, ", lambda_beta=", x$lambda_beta_used, "\n")

  if (!is.null(x$convergence_info)) {
    cat("Converged:", ifelse(x$convergence_info$converged, "Yes", "No"), "\n")
  }
}

#' Backward compatible runTransCox_one function
#'
#' @description
#' A wrapper for backward compatibility with previous versions.
#'
#' @param Pout Primary parameters output.
#' @param l1 L1 penalty for eta.
#' @param l2 L1 penalty for xi.
#' @param learning_rate Learning rate.
#' @param nsteps Number of steps.
#' @param cov Covariate names.
#' @param lambda_beta L1 penalty for beta.
#' @param use_sparse Logical.
#'
#' @export
runTransCox_one <- function(Pout, l1 = 1, l2 = 1, learning_rate = 0.004, nsteps = 200,
                            cov = c('X1', 'X2'), lambda_beta = 0, use_sparse = NULL) {

  # Automatically detect sparse version usage
  if (is.null(use_sparse)) {
    n_features <- length(cov)
    has_lambda_beta <- (length(lambda_beta) > 1) || (length(lambda_beta) == 1 && lambda_beta > 0)
    use_sparse <- (n_features > 50 || has_lambda_beta)
  }

  # Check if using sparse version
  use_sparse_version <- use_sparse && (
    (length(lambda_beta) > 1) ||
      (length(lambda_beta) == 1 && lambda_beta > 0)
  )

  if (use_sparse_version) {
    if (!exists("TransCox_Sparse", mode = "function")) {
      py_path <- system.file("python", "TransCoxFunction_Sparse.py", package = "TransCoxSparse")
      if (py_path == "") py_path <- file.path(getwd(), "inst", "python", "TransCoxFunction_Sparse.py")
      if (file.exists(py_path)) reticulate::source_python(py_path, envir = environment())
    }

    CovData <- Pout$primData[, cov]
    status <- Pout$primData[, "status"]
    cumH <- Pout$primData$fullCumQ
    hazards <- Pout$dQ$dQ

    test <- TransCox_Sparse(CovData = as.matrix(CovData),
                            cumH = cumH,
                            hazards = hazards,
                            status = status,
                            estR = Pout$estR,
                            Xinn = Pout$Xinn,
                            lambda1 = l1, lambda2 = l2, lambda_beta = lambda_beta,
                            learning_rate = learning_rate,
                            nsteps = nsteps,
                            verbose = FALSE)

    return(list(eta = test[[1]],
                xi = test[[2]],
                new_beta = test[[3]],
                new_IntH = Pout$dQ$dQ + test[[2]],
                time = Pout$primData[status == 2, "time"]))

  } else {
    if (!exists("TransCox", mode = "function")) {
      py_path <- system.file("python", "TransCoxFunction.py", package = "TransCoxSparse")
      if (py_path == "") py_path <- file.path(getwd(), "inst", "python", "TransCoxFunction.py")
      if (file.exists(py_path)) reticulate::source_python(py_path, envir = environment())
    }

    CovData = Pout$primData[, cov]
    status = Pout$primData[, "status"]
    cumH = Pout$primData$fullCumQ
    hazards = Pout$dQ$dQ

    test <- TransCox(CovData = as.matrix(CovData),
                     cumH = cumH,
                     hazards = hazards,
                     status = status,
                     estR = Pout$estR,
                     Xinn = Pout$Xinn,
                     lambda1 = l1, lambda2 = l2,
                     learning_rate = learning_rate,
                     nsteps = nsteps)
    names(test) <- c("eta", "xi")

    return(list(eta = test$eta,
                xi = test$xi,
                new_beta = Pout$estR + test$eta,
                new_IntH = Pout$dQ$dQ + test$xi,
                time = Pout$primData[status == 2, "time"]))
  }
}
