#' Helper function: generate fine search vector
#'
#' @param best_value The optimal value found in the coarse search.
#' @param coarse_vec The original vector used for the coarse search.
#'
#' @return A numeric vector for fine search.
#' @noRd
generate_fine_search_vector <- function(best_value, coarse_vec) {
  # [SCIENTIFIC FIX] Ensure vector is sorted to make neighbor logic valid
  coarse_vec <- sort(unique(coarse_vec))

  idx <- which.min(abs(coarse_vec - best_value))

  if (idx == 1) {
    lower <- best_value
    # Use next value or heuristic upper bound
    upper <- if (length(coarse_vec) > 1) coarse_vec[2] else best_value * 2
  } else if (idx == length(coarse_vec)) {
    lower <- coarse_vec[idx - 1]
    upper <- best_value
  } else {
    lower <- coarse_vec[idx - 1]
    upper <- coarse_vec[idx + 1]
  }

  # Generate 6 points
  fine_vec <- seq(lower, upper, length.out = 6)
  fine_vec <- sort(unique(fine_vec))
  return(fine_vec)
}

#' BIC-Based Parameter Selection for Sparse TransCox (PGD Version)
#'
#' @description
#' Performs BIC-based parameter selection. Adapted for Proximal Gradient Descent:
#' Lambda ranges are adjusted for normalized likelihood, and threshold_c is removed.
#'
#' @param primData Target domain data.
#' @param auxData Source domain data.
#' @param cov Covariate names.
#' @param statusvar Status variable name.
#' @param lambda1_vec Candidate values for Transfer Penalty.
#' @param lambda2_vec Candidate values for Baseline Hazard Penalty.
#' @param lambda_beta_vec Candidate values for Global Sparsity.
#' @param learning_rate Learning rate (Default 0.01 for PGD).
#' @param nsteps Optimization steps (Default 2000 for PGD).
#' @param use_sparse Logical. Always TRUE.
#' @param parallel Logical. Enable parallel computing.
#' @param verbose Logical. Print progress.
#' @param n_cores Integer. Number of cores.
#'
#' @return List of optimal parameters and model results.
#' @export
SelParam_By_BIC_Sparse <- function(primData, auxData, cov = c("X1", "X2"),
                                   statusvar = "status",
                                   # [SCIENTIFIC FIX] Adjusted ranges for Normalized Loss (1/N)
                                   # Old range 5.0 is too large for normalized loss (~1.0)
                                   lambda1_vec = c(0.01, 0.05, 0.1, 0.2, 0.5),
                                   lambda2_vec = c(0.001, 0.01, 0.05, 0.1),
                                   lambda_beta_vec = c(0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
                                   learning_rate = 0.01,
                                   nsteps = 2000,
                                   use_sparse = TRUE,
                                   parallel = FALSE,
                                   verbose = TRUE,
                                   n_cores = NULL) { # threshold_c REMOVED

  if (verbose) cat("Starting BIC parameter selection (PGD Optimized)...\n")

  result_cache <- new.env()
  cl <- NULL

  # Parallel computation initialization
  if (parallel) {
    if (is.null(n_cores)) {
      n_cores <- max(1, parallel::detectCores() - 1)
    }
    n_cores <- min(n_cores, parallel::detectCores())

    if (verbose) cat(sprintf("Parallel computing enabled, using %d cores\n", n_cores))

    if (!requireNamespace("parallel", quietly = TRUE) || !requireNamespace("doParallel", quietly = TRUE)) {
      if (verbose) cat("Parallel packages not found, switching to serial mode\n")
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
  if (!exists("TransCox_Sparse")) {
    py_file <- system.file("python", "TransCoxFunction_Sparse.py", package = "TransCoxSparse")
    if (py_file == "") py_file <- file.path(getwd(), "inst", "python", "TransCoxFunction_Sparse.py")
    if (file.exists(py_file)) reticulate::source_python(py_file)
  }

  # Estimate source domain parameters
  Cout <- GetAuxSurv_Sparse(auxData, cov = cov)
  Pout <- GetPrimaryParam(primData, q = Cout$q, estR = Cout$estR)

  if (verbose) cat("Pre-calculating data structures...\n")

  CovData_precomputed <- data.matrix(Pout$primData[, cov, drop = FALSE])
  status_precomputed <- as.integer(Pout$primData[, statusvar])
  estR_precomputed <- as.double(Pout$estR)
  Xinn_precomputed <- data.matrix(Pout$Xinn)
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
    lambda1 = sort(unique(lambda1_vec)),
    lambda2 = sort(unique(lambda2_vec)),
    lambda_beta = sort(unique(lambda_beta_vec)),
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
  # [FIX] threshold_c REMOVED from arguments
  evaluate_params <- function(idx, param_row, data_pkg, use_sparse_flag, lr, n_steps, verb = FALSE) {
    lambda1 <- param_row$lambda1
    lambda2 <- param_row$lambda2
    lambda_beta <- param_row$lambda_beta

    if (lambda1 <= 0 || lambda2 <= 0 || (use_sparse_flag && lambda_beta < 0)) {
      return(list(bic = Inf, success = FALSE))
    }

    tryCatch({
      if (use_sparse_flag && !is.null(lambda_beta) && lambda_beta > 0) {
        # [FIX] threshold_c REMOVED from Python call
        result <- TransCox_Sparse(
          CovData = data_pkg$CovData,
          cumH = data_pkg$cumH,
          hazards = data_pkg$hazards,
          status = data_pkg$status,
          estR = data_pkg$estR,
          Xinn = data_pkg$Xinn,
          lambda1 = as.double(lambda1),
          lambda2 = as.double(lambda2),
          lambda_beta = as.double(lambda_beta),
          learning_rate = as.double(lr),
          nsteps = as.integer(n_steps),
          verbose = FALSE
        )
        eta <- result[[1]]
        xi <- result[[2]]
        newBeta <- result[[3]]
      } else {
        stop("Non-sparse TransCox not supported in this version.")
      }

      # [SCIENTIFIC FIX] Additive Hazard: h_new = h_old + xi
      newHaz <- data_pkg$hazards + xi

      if (any(is.na(eta)) || any(is.na(xi)) || any(is.na(newBeta))) {
        return(list(bic = Inf, success = FALSE, error = "NaNs produced"))
      }

      bic_value <- GetBIC(
        status = data_pkg$status,
        CovData = data_pkg$CovData,
        hazards = data_pkg$hazards,
        newBeta = newBeta,
        newHaz = newHaz,      # Corrected Additive Hazard
        eta = eta,
        xi = xi,
        cutoff = 1e-8,        # PGD produces exact zeros
        lambda1 = lambda1,
        lambda2 = lambda2,
        lambda_beta = lambda_beta
      )

      if (is.na(bic_value) || is.infinite(bic_value)) {
        return(list(bic = Inf, success = FALSE))
      }

      return(list(
        idx = idx,
        lambda1 = lambda1,
        lambda2 = lambda2,
        lambda_beta = lambda_beta,
        bic = bic_value,
        success = TRUE
      ))

    }, error = function(e) {
      return(list(bic = Inf, success = FALSE, error = e$message))
    })
  }

  # Execute parameter search
  results <- NULL
  parallel_success <- FALSE

  if (parallel && !is.null(cl)) {
    if (verbose) cat("Starting parallel parameter search...\n")

    tryCatch({
      py_path <- Sys.getenv("RETICULATE_PYTHON")
      parallel::clusterExport(cl, varlist = c("py_path"), envir = environment())

      parallel::clusterEvalQ(cl, {
        library(reticulate)
        if (nzchar(Sys.getenv("RETICULATE_PYTHON"))) {
          reticulate::use_python(Sys.getenv("RETICULATE_PYTHON"), required = TRUE)
        }
        py_file <- system.file("python", "TransCoxFunction_Sparse.py", package = "TransCoxSparse")
        if (py_file == "") py_file <- "inst/python/TransCoxFunction_Sparse.py"
        if (file.exists(py_file)) reticulate::source_python(py_file)
      })

      parallel::clusterExport(cl, c("GetBIC", "evaluate_params"), envir = environment())

      # [FIX] threshold_c REMOVED from parLapply call
      results <- parallel::parLapply(cl, 1:nrow(param_grid), function(i) {
        evaluate_params(i, param_grid[i, ], data_package, use_sparse, learning_rate, nsteps, FALSE)
      })

      parallel_success <- TRUE

    }, error = function(e) {
      if (verbose) cat("Parallel computation failed:", e$message, "\nSwitching to serial.\n")
      results <<- NULL
    })
  }

  if (!parallel_success || is.null(results)) {
    if (verbose) cat("Starting serial parameter search...\n")
    results <- lapply(1:nrow(param_grid), function(i) {
      # [FIX] threshold_c REMOVED from serial call
      result <- evaluate_params(i, param_grid[i, ], data_package, use_sparse, learning_rate, nsteps, FALSE)
      if (verbose && i %% 5 == 0) cat(sprintf("Progress: %d/%d\n", i, nrow(param_grid)))
      return(result)
    })
  }

  # Process Results
  BIC_array <- array(Inf, dim = c(length(lambda1_vec), length(lambda2_vec), length(lambda_beta_vec)),
                     dimnames = list(lambda1_vec, lambda2_vec, lambda_beta_vec))

  for (result in results) {
    if (result$success) {
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
    cat(sprintf("Coarse search completed, optimal: L1=%.4f, L2=%.4f, Lbeta=%.4f, BIC=%.3f\n",
                best_lambda1, best_lambda2, best_lambda_beta, current_best_bic))
  }

  # --- Phase 2: Fine Search ---
  if (verbose) cat("Phase 2: Fine search...\n")

  fine_l1 <- generate_fine_search_vector(best_lambda1, lambda1_vec)
  fine_l2 <- generate_fine_search_vector(best_lambda2, lambda2_vec)
  fine_lb <- generate_fine_search_vector(best_lambda_beta, lambda_beta_vec)

  fine_grid <- expand.grid(lambda1 = fine_l1, lambda2 = fine_l2, lambda_beta = fine_lb, stringsAsFactors = FALSE)

  # Filter duplicates roughly
  fine_grid <- fine_grid[!paste(fine_grid$lambda1, fine_grid$lambda2, fine_grid$lambda_beta) %in%
                           paste(coarse_grid$lambda1, coarse_grid$lambda2, coarse_grid$lambda_beta), ]

  if (nrow(fine_grid) > 0) {
    if (verbose) cat(sprintf("Fine search adding %d parameter combinations...\n", nrow(fine_grid)))

    fine_results <- lapply(1:nrow(fine_grid), function(i) {
      # [FIX] threshold_c REMOVED
      evaluate_params(i, fine_grid[i, ], data_package, use_sparse, learning_rate, nsteps, FALSE)
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

  # --- Final Run ---
  # [FIX] threshold_c REMOVED
  final_result <- TransCox_Sparse(
    CovData = data_package$CovData,
    cumH = data_package$cumH,
    hazards = data_package$hazards,
    status = data_package$status,
    estR = data_package$estR,
    Xinn = data_package$Xinn,
    lambda1 = as.double(best_lambda1),
    lambda2 = as.double(best_lambda2),
    lambda_beta = as.double(best_lambda_beta),
    learning_rate = as.double(learning_rate),
    nsteps = as.integer(nsteps),
    verbose = verbose
  )

  return(list(
    best_lambda1 = best_lambda1,
    best_lambda2 = best_lambda2,
    best_lambda_beta = best_lambda_beta,
    BIC_array = BIC_array,
    final_eta = final_result[[1]],
    final_xi = final_result[[2]],
    final_beta = final_result[[3]],
    convergence_info = final_result[[4]],
    source_estR = data_package$estR,
    base_haz = hazards_precomputed
  ))
}
