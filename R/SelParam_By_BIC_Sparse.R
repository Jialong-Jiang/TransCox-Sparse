#' Helper function: generate fine search vector
#'
#' @param best_value The optimal value found in the coarse search.
#' @param coarse_vec The original vector used for the coarse search.
#'
#' @return A numeric vector for fine search.
#' @noRd
generate_fine_search_vector <- function(best_value, coarse_vec) {
  idx <- which.min(abs(coarse_vec - best_value))

  if (idx == 1) {
    lower <- best_value
    upper <- if (length(coarse_vec) > 1) coarse_vec[2] else best_value * 2
  } else if (idx == length(coarse_vec)) {
    lower <- coarse_vec[idx - 1]
    upper <- best_value
  } else {
    lower <- coarse_vec[idx - 1]
    upper <- coarse_vec[idx + 1]
  }

  fine_vec <- seq(lower, upper, length.out = 6)
  fine_vec <- sort(unique(fine_vec))
  return(fine_vec)
}

#' BIC-Based Parameter Selection for Sparse TransCox
#'
#' @description
#' Performs Bayesian Information Criterion (BIC) based parameter selection for
#' the sparse TransCox model. This function extends the original BIC selection
#' to support three-dimensional parameter tuning including lambda_beta for
#' target domain sparsity control.
#'
#' @details
#' This function conducts a comprehensive grid search over three regularization
#' parameters (lambda1, lambda2, lambda_beta) to find the optimal combination
#' that minimizes the BIC. The search can be parallelized for computational
#' efficiency when dealing with large parameter grids.
#'
#' @param primData A data.frame containing the target domain survival data.
#' @param auxData A data.frame containing the source domain survival data.
#' @param cov A character vector specifying the names of covariates to be used.
#'    Default is c("X1", "X2").
#' @param statusvar A character string specifying the name of the event status
#'    variable. Default is "status".
#' @param lambda1_vec Numeric vector of candidate values for lambda1 (L1 penalty
#'    for eta parameter). Default provides a reasonable range.
#' @param lambda2_vec Numeric vector of candidate values for lambda2 (L1 penalty
#'    for xi parameter). Default provides a reasonable range.
#' @param lambda_beta_vec Numeric vector of candidate values for lambda_beta
#'    (L1 penalty for beta_t parameter). Default includes zero for non-sparse option.
#' @param learning_rate Numeric. Learning rate for the optimization algorithm.
#'    Default is 0.004.
#' @param nsteps Integer. Maximum number of optimization steps. Default is 200.
#' @param use_sparse Logical. Whether to use sparse implementation. Default is TRUE.
#' @param parallel Logical. Whether to use parallel computation for grid search.
#'    Default is FALSE.
#' @param verbose Logical. Whether to display detailed progress information.
#'    Default is TRUE.
#' @param n_cores Integer. Number of cores for parallel computation. If NULL, detected automatically.
#' @param threshold_c Numeric. Constant for theoretical hard thresholding (tau = C * sqrt(log(p)/n)).
#'    Default is 0.5.
#'
#' @return A list containing the following components:
#' \describe{
#'    \item{optimal_lambda1}{The optimal lambda1 value that minimizes BIC}
#'    \item{optimal_lambda2}{The optimal lambda2 value that minimizes BIC}
#'    \item{optimal_lambda_beta}{The optimal lambda_beta value that minimizes BIC}
#'    \item{min_bic}{The minimum BIC value achieved}
#'    \item{bic_matrix}{3D array of BIC values for all parameter combinations}
#'    \item{parameter_grid}{Data frame of all tested parameter combinations}
#'    \item{convergence_info}{Information about optimization convergence}
#' }
#'
#' @importFrom parallel makeCluster stopCluster detectCores clusterEvalQ clusterExport parLapply
#' @importFrom doParallel registerDoParallel
#' @importFrom reticulate source_python use_python use_condaenv py_config
#' @export
SelParam_By_BIC_Sparse <- function(primData, auxData, cov = c("X1", "X2"),
                                   statusvar = "status",
                                   lambda1_vec = c(0, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0),
                                   lambda2_vec = c(0.001, 0.01, 0.1, 0.5, 1.0, 2.0),
                                   lambda_beta_vec = c(0, 0.0001, 0.0005, 0.001, 0.01, 0.1, 0.5, 1.0),
                                   learning_rate = 0.001,
                                   nsteps = 500,
                                   use_sparse = TRUE,
                                   parallel = FALSE,
                                   verbose = TRUE,
                                   n_cores = NULL,
                                   threshold_c = 0.5) { # [NEW] Added parameter

  if (verbose) cat("Starting BIC parameter selection...\n")

  result_cache <- new.env()
  cl <- NULL

  # Parallel computation initialization
  if (parallel) {
    if (is.null(n_cores)) {
      n_cores <- max(1, parallel::detectCores() - 1)
    }
    n_cores <- min(n_cores, parallel::detectCores())

    if (verbose) cat(sprintf("Parallel computing enabled, using %d cores\n", n_cores))

    if (!requireNamespace("parallel", quietly = TRUE)) {
      if (verbose) cat("Package 'parallel' not available, switching to serial mode\n")
      parallel <- FALSE
    } else if (!requireNamespace("doParallel", quietly = TRUE)) {
      if (verbose) cat("Package 'doParallel' not available, switching to serial mode\n")
      parallel <- FALSE
    } else {
      cl <- parallel::makeCluster(n_cores)
      doParallel::registerDoParallel(cl)
      on.exit({
        if (!is.null(cl)) {
          parallel::stopCluster(cl)
          if (verbose) cat("Parallel cluster cleaned up\n")
        }
      }, add = TRUE)
    }
  }

  # Load Python functions
  if (use_sparse) {
    if (!exists("TransCox_Sparse")) {
      py_file <- system.file("python", "TransCoxFunction_Sparse.py", package = "TransCoxSparse")
      if (py_file == "") py_file <- file.path(getwd(), "inst", "python", "TransCoxFunction_Sparse.py")
      if (file.exists(py_file)) reticulate::source_python(py_file)
    }
  } else {
    if (!exists("TransCox")) {
      py_file <- system.file("python", "TransCoxFunction.py", package = "TransCoxSparse")
      if (py_file == "") py_file <- file.path(getwd(), "inst", "python", "TransCoxFunction.py")
      if (file.exists(py_file)) reticulate::source_python(py_file)
    }
  }

  # Estimate source domain parameters
  if (use_sparse) {
    Cout <- GetAuxSurv_Sparse(auxData, cov = cov)
  } else {
    Cout <- GetAuxSurv(auxData, cov = cov)
  }

  # Calculate target domain parameters
  Pout <- GetPrimaryParam(primData, q = Cout$q, estR = Cout$estR)

  if (verbose) cat("Pre-calculating data to optimize performance...\n")

  CovData_precomputed <- as.matrix(Pout$primData[, cov, drop = FALSE])
  storage.mode(CovData_precomputed) <- "double"
  status_precomputed <- as.integer(Pout$primData[, statusvar])

  estR_precomputed <- as.double(Pout$estR)
  Xinn_precomputed <- as.matrix(Pout$Xinn)
  storage.mode(Xinn_precomputed) <- "double"
  cumH_precomputed <- as.double(Pout$primData$fullCumQ)
  hazards_precomputed <- as.double(Pout$dQ$dQ)

  data_package <- list(
    CovData = CovData_precomputed,
    cumH = cumH_precomputed,
    hazards = hazards_precomputed,
    status = status_precomputed,
    estR = estR_precomputed,
    Xinn = Xinn_precomputed
  )

  if (verbose) cat("Starting hierarchical parameter search...\n")

  coarse_grid <- expand.grid(
    lambda1 = lambda1_vec,
    lambda2 = lambda2_vec,
    lambda_beta = lambda_beta_vec,
    stringsAsFactors = FALSE
  )

  if (verbose) {
    cat(sprintf("Phase 1: Coarse search, %d parameter combinations...\n", nrow(coarse_grid)))
  }

  param_grid <- coarse_grid
  bic_cache <- list()

  find_cached_result <- function(lambda1, lambda2, lambda_beta) {
    for (cached in bic_cache) {
      if (abs(cached$lambda1 - lambda1) < 1e-6 &&
          abs(cached$lambda2 - lambda2) < 1e-6 &&
          abs(cached$lambda_beta - lambda_beta) < 1e-6) {
        return(cached)
      }
    }
    return(NULL)
  }

  # Define evaluation function
  # [CRITICAL UPDATE] Added thresh_c argument
  evaluate_params <- function(idx, param_row, data_pkg, use_sparse_flag, lr, n_steps, thresh_c, verb = FALSE) {
    lambda1 <- param_row$lambda1
    lambda2 <- param_row$lambda2
    lambda_beta <- param_row$lambda_beta

    cached_result <- find_cached_result(lambda1, lambda2, lambda_beta)
    if (!is.null(cached_result)) {
      return(list(
        idx = idx, lambda1 = lambda1, lambda2 = lambda2, lambda_beta = lambda_beta,
        bic = cached_result$bic, success = TRUE, cached = TRUE
      ))
    }

    if (lambda1 <= 0 || lambda2 <= 0 || (use_sparse_flag && lambda_beta < 0)) {
      return(list(
        lambda1 = lambda1, lambda2 = lambda2, lambda_beta = lambda_beta,
        bic = Inf, success = FALSE, early_stop = TRUE
      ))
    }

    tryCatch({
      if (use_sparse_flag && !is.null(lambda_beta) && lambda_beta > 0) {
        # Use sparse version
        # [CRITICAL UPDATE] Passing threshold_c to Python
        result <- TransCox_Sparse(
          CovData = data_pkg$CovData,
          cumH = data_pkg$cumH,
          hazards = data_pkg$hazards,
          status = data_pkg$status,
          estR = data_pkg$estR,
          Xinn = data_pkg$Xinn,
          lambda1 = lambda1,
          lambda2 = lambda2,
          lambda_beta = lambda_beta,
          learning_rate = lr,
          nsteps = n_steps,
          verbose = FALSE,
          threshold_c = as.double(thresh_c) # Pass C value
        )
        eta <- result[[1]]
        xi <- result[[2]]
        newBeta <- result[[3]]
      } else {
        # Use original version
        test <- TransCox(
          CovData = data_pkg$CovData,
          cumH = data_pkg$cumH,
          hazards = data_pkg$hazards,
          status = data_pkg$status,
          estR = data_pkg$estR,
          Xinn = data_pkg$Xinn,
          lambda1 = lambda1,
          lambda2 = lambda2,
          learning_rate = lr,
          nsteps = n_steps
        )
        eta <- test[[1]]
        xi <- test[[2]]
        newBeta <- data_pkg$estR + eta
      }

      newHaz <- data_pkg$hazards + xi

      if (any(is.na(eta)) || any(is.na(xi)) || any(is.na(newBeta)) ||
          any(is.infinite(eta)) || any(is.infinite(xi)) || any(is.infinite(newBeta))) {
        return(list(
          idx = idx, lambda1 = lambda1, lambda2 = lambda2, lambda_beta = lambda_beta,
          bic = Inf, success = FALSE, early_stop = TRUE,
          error = "Invalid results (NA or Inf values)"
        ))
      }

      bic_value <- GetBIC(
        status = data_pkg$status,
        CovData = data_pkg$CovData,
        hazards = data_pkg$hazards,
        newBeta = newBeta,
        newHaz = newHaz,
        eta = eta,
        xi = xi,
        cutoff = 1e-5,
        lambda1 = lambda1,
        lambda2 = lambda2,
        lambda_beta = lambda_beta
      )

      if (is.na(bic_value) || is.infinite(bic_value) || bic_value > 1e6) {
        return(list(
          idx = idx, lambda1 = lambda1, lambda2 = lambda2, lambda_beta = lambda_beta,
          bic = Inf, success = FALSE, early_stop = TRUE,
          error = "Invalid BIC value"
        ))
      }

      bic_cache <<- append(bic_cache, list(list(
        lambda1 = lambda1, lambda2 = lambda2, lambda_beta = lambda_beta,
        bic = bic_value
      )))

      return(list(
        idx = idx,
        lambda1 = lambda1,
        lambda2 = lambda2,
        lambda_beta = lambda_beta,
        bic = bic_value,
        eta = eta,
        xi = xi,
        newBeta = newBeta,
        success = TRUE
      ))

    }, error = function(e) {
      return(list(
        idx = idx,
        lambda1 = lambda1,
        lambda2 = lambda2,
        lambda_beta = lambda_beta,
        bic = Inf,
        success = FALSE,
        error = e$message
      ))
    })
  }

  # Execute parameter search
  results <- NULL
  parallel_success <- FALSE

  if (parallel && !is.null(cl)) {
    if (verbose) cat("Starting parallel parameter search...\n")

    tryCatch({
      py_path <- Sys.getenv("RETICULATE_PYTHON")
      conda_env <- Sys.getenv("RETICULATE_CONDA_ENV")

      parallel::clusterExport(cl, varlist = c("py_path", "conda_env"), envir = environment())

      parallel::clusterEvalQ(cl, {
        if (nzchar(py_path)) Sys.setenv(RETICULATE_PYTHON = py_path)
        if (nzchar(conda_env)) Sys.setenv(RETICULATE_CONDA_ENV = conda_env)

        library(reticulate)
        if (nzchar(Sys.getenv("RETICULATE_PYTHON"))) {
          reticulate::use_python(Sys.getenv("RETICULATE_PYTHON"), required = TRUE)
        }
        if (nzchar(Sys.getenv("RETICULATE_CONDA_ENV"))) {
          reticulate::use_condaenv(Sys.getenv("RETICULATE_CONDA_ENV"))
        }

        py_file_sparse <- system.file("python", "TransCoxFunction_Sparse.py", package = "TransCoxSparse")
        if (py_file_sparse == "") py_file_sparse <- "inst/python/TransCoxFunction_Sparse.py"
        if (file.exists(py_file_sparse)) reticulate::source_python(py_file_sparse)

        py_file <- system.file("python", "TransCoxFunction.py", package = "TransCoxSparse")
        if (py_file == "") py_file <- "inst/python/TransCoxFunction.py"
        if (file.exists(py_file)) reticulate::source_python(py_file)
      })

      parallel::clusterExport(cl, c("GetBIC", "evaluate_params"), envir = environment())

      # [CRITICAL UPDATE] Pass threshold_c to parLapply
      results <- parallel::parLapply(cl, 1:nrow(param_grid), function(i) {
        evaluate_params(i, param_grid[i, ], data_package, use_sparse, learning_rate, nsteps, threshold_c, FALSE)
      })

      parallel_success <- TRUE

    }, error = function(e) {
      if (verbose) cat("Parallel computation failed, switching to serial mode:", e$message, "\n")
      results <<- NULL
    })
  }

  if (!parallel_success || is.null(results)) {
    if (verbose) cat("Starting serial parameter search...\n")
    results <- lapply(1:nrow(param_grid), function(i) {
      # [CRITICAL UPDATE] Pass threshold_c to serial lapply
      result <- evaluate_params(i, param_grid[i, ], data_package, use_sparse, learning_rate, nsteps, threshold_c, FALSE)
      if (verbose && i %% 5 == 0) {
        progress <- i / nrow(param_grid) * 100
        cat(sprintf("Progress: %.1f%% (%d/%d)\n", progress, i, nrow(param_grid)))
      }
      return(result)
    })
  }

  BIC_array <- array(Inf, dim = c(length(lambda1_vec), length(lambda2_vec), length(lambda_beta_vec)),
                     dimnames = list(lambda1_vec, lambda2_vec, lambda_beta_vec))

  for (result in results) {
    if (!is.null(result$bic)) {
      i <- which(abs(lambda1_vec - result$lambda1) < 1e-6)
      j <- which(abs(lambda2_vec - result$lambda2) < 1e-6)
      k <- which(abs(lambda_beta_vec - result$lambda_beta) < 1e-6)
      if (length(i) > 0 && length(j) > 0 && length(k) > 0) {
        BIC_array[i, j, k] <- result$bic
      }
    }
  }

  min_idx <- which(BIC_array == min(BIC_array, na.rm = TRUE), arr.ind = TRUE)
  if (nrow(min_idx) > 1) min_idx <- min_idx[1, , drop = FALSE]

  best_lambda1 <- lambda1_vec[min_idx[1]]
  best_lambda2 <- lambda2_vec[min_idx[2]]
  best_lambda_beta <- lambda_beta_vec[min_idx[3]]
  current_best_bic <- min(BIC_array, na.rm = TRUE)

  if (verbose) {
    cat(sprintf("Coarse search completed, optimal: L1=%.3f, L2=%.3f, Lbeta=%.4f, BIC=%.3f\n",
                best_lambda1, best_lambda2, best_lambda_beta, current_best_bic))
  }

  if (verbose) cat("Phase 2: Fine search around optimal area...\n")

  fine_lambda1_vec <- generate_fine_search_vector(best_lambda1, lambda1_vec)
  fine_lambda2_vec <- generate_fine_search_vector(best_lambda2, lambda2_vec)
  fine_lambda_beta_vec <- generate_fine_search_vector(best_lambda_beta, lambda_beta_vec)

  fine_grid <- expand.grid(
    lambda1 = fine_lambda1_vec,
    lambda2 = fine_lambda2_vec,
    lambda_beta = fine_lambda_beta_vec,
    stringsAsFactors = FALSE
  )

  fine_grid <- fine_grid[!paste(fine_grid$lambda1, fine_grid$lambda2, fine_grid$lambda_beta) %in%
                           paste(coarse_grid$lambda1, coarse_grid$lambda2, coarse_grid$lambda_beta), ]

  if (nrow(fine_grid) > 0) {
    if (verbose) cat(sprintf("Fine search adding %d parameter combinations...\n", nrow(fine_grid)))

    fine_results <- lapply(1:nrow(fine_grid), function(i) {
      # [CRITICAL UPDATE] Pass threshold_c to fine search
      evaluate_params(i, fine_grid[i, ], data_package, use_sparse, learning_rate, nsteps, threshold_c, FALSE)
    })

    fine_BIC_values <- sapply(fine_results, function(x) x$bic)
    min_fine_bic <- min(fine_BIC_values, na.rm = TRUE)

    if (min_fine_bic < current_best_bic) {
      min_fine_idx <- which.min(fine_BIC_values)
      best_fine_result <- fine_results[[min_fine_idx]]
      best_lambda1 <- best_fine_result$lambda1
      best_lambda2 <- best_fine_result$lambda2
      best_lambda_beta <- best_fine_result$lambda_beta
      if (verbose) cat(sprintf("Fine search found better parameters: BIC=%.3f\n", min_fine_bic))
    }
  }

  if (verbose) cat("Hierarchical search completed.\n")

  # Final run
  if (use_sparse && best_lambda_beta > 0) {
    # [CRITICAL UPDATE] Pass threshold_c to final model
    final_result <- TransCox_Sparse(
      CovData = as.matrix(Pout$primData[, cov]),
      cumH = cumH_precomputed,
      hazards = hazards_precomputed,
      status = status_precomputed,
      estR = estR_precomputed,
      Xinn = Xinn_precomputed,
      lambda1 = best_lambda1,
      lambda2 = best_lambda2,
      lambda_beta = best_lambda_beta,
      learning_rate = learning_rate,
      nsteps = nsteps,
      verbose = verbose,
      threshold_c = as.double(threshold_c) # Pass C value
    )
    final_eta <- final_result[[1]]
    final_xi <- final_result[[2]]
    final_beta <- final_result[[3]]
    convergence_info <- final_result[[4]]
  } else {
    final_test <- TransCox(
      CovData = as.matrix(Pout$primData[, cov]),
      cumH = cumH_precomputed,
      hazards = hazards_precomputed,
      status = status_precomputed,
      estR = estR_precomputed,
      Xinn = Xinn_precomputed,
      lambda1 = best_lambda1,
      lambda2 = best_lambda2,
      learning_rate = learning_rate,
      nsteps = nsteps
    )
    final_eta <- final_test[[1]]
    final_xi <- final_test[[2]]
    final_beta <- Pout$estR + final_eta
    convergence_info <- NULL
  }

  return(list(
    best_lambda1 = best_lambda1,
    best_lambda2 = best_lambda2,
    best_lambda_beta = best_lambda_beta,
    BIC_array = BIC_array,
    final_eta = final_eta,
    final_xi = final_xi,
    final_beta = final_beta,
    convergence_info = convergence_info,
    source_estR = Pout$estR,
    lambda1_vec = lambda1_vec,
    lambda2_vec = lambda2_vec,
    lambda_beta_vec = lambda_beta_vec
  ))
}
