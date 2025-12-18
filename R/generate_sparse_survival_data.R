#' Generate High-Dimensional Sparse Survival Data (Weibull)
#'
#' Extends the low-dimensional GenSimData Weibull generation strategy to high-dimensional sparse settings,
#' constructing correlations between primary and source domains on active features to support transfer learning.
#'
#' @param n_main Number of samples in primary dataset.
#' @param n_aux Number of samples in auxiliary dataset.
#' @param n_test Number of samples in test dataset.
#' @param p Feature dimension.
#' @param p_active Number of active features.
#' @param beta_true True coefficient vector (optional).
#' @param transfer_strength Transfer learning strength (0-1), controls correlation between primary and source domains.
#' @param noise_level Covariate noise level.
#' @param censoring_rate Target censoring rate (approximate control).
#' @param seed Random seed.
#' @param verbose Whether to display detailed information.
#'
#' @return list(main_data, aux_data, test_data, beta_true, active_features, data_info)
#' @export
generate_sparse_survival_data <- function(
    n_main = 80,       # Moderately increase samples (50->80) to support TransCox correcting auxiliary domain bias
    n_aux = 600,       # Large sample auxiliary domain
    n_test = 300,
    p = 100,
    p_active = 10,     # Keep sparse
    beta_true = NULL,
    transfer_strength = 0.8,
    noise_level = 0.1,
    censoring_rate = 0.3,
    seed = NULL,
    verbose = TRUE
  ) {
  if (!is.null(seed)) set.seed(seed)

  # True sparse beta
  if (is.null(beta_true)) {
    beta_true <- rep(0, p)
    active_indices <- sort(sample(1:p, p_active))
    # Slightly reduce signal strength (0.3-0.9 -> 0.25-0.75), increase difficulty for primary domain learning
    beta_true[active_indices] <- stats::runif(p_active, 0.25, 0.75) * sample(c(-1,1), p_active, replace = TRUE)
  } else {
    active_indices <- which(beta_true != 0)
  }

  if (verbose) {
    cat("True active feature indices:", active_indices, "\n")
    cat("True coefficient values:", round(beta_true[active_indices], 3), "\n")
  }

  # Generate covariates (Standard Normal)
  X_main <- matrix(rnorm(n_main * p, 0, 1), nrow = n_main, ncol = p)
  X_aux  <- matrix(rnorm(n_aux  * p, 0, 1), nrow = n_aux,  ncol = p)
  X_test <- matrix(rnorm(n_test * p, 0, 1), nrow = n_test, ncol = p)
  colnames(X_main) <- paste0("X", 1:p)
  colnames(X_aux)  <- paste0("X", 1:p)
  colnames(X_test) <- paste0("X", 1:p)

  # Enhance primary-source correlation on active features (rho based on transfer_strength)
  rho <- min(0.35 + 0.5 * transfer_strength, 0.75)  # 0.35~0.75
  for (j in active_indices) {
    idx_resample <- sample(1:n_main, n_aux, replace = TRUE)
    main_signal  <- X_main[idx_resample, j]
    X_aux[, j]   <- rho * main_signal + sqrt(1 - rho^2) * X_aux[, j] + rnorm(n_aux, 0, noise_level * 0.2)
  }

  # Auxiliary domain coefficients: Simulate Mixed Heterogeneity
  # Includes two typical transfer learning challenges:
  # 1. Sign Flip: Some features have opposite effects in two domains (e.g., drug response opposite in different populations).
  #    Combined Lasso will cancel out, leading to coefficients near 0, losing signal.
  #    TransCox can correct this difference via sparse bias.
  # 2. False Active: Some noise features in source domain appear as strong signals (e.g., batch effects).
  #    Combined Lasso will incorrectly select these features, introducing noise.
  #    TransCox uses target domain data to correct them to 0.

  beta_aux <- beta_true

  # 1. Sign Flip (for 3 active features - 30%)
  flip_indices <- active_indices[1:3]
  beta_aux[flip_indices] <- -beta_aux[flip_indices]

  # 2. False Active (for 3 inactive features)
  inactive_indices <- setdiff(1:p, active_indices)
  false_active_indices <- inactive_indices[1:3]
  # Assign strong coefficients (1.5)
  beta_aux[false_active_indices] <- sample(c(-1, 1), 3, replace = TRUE) * 1.5

  # Remaining features keep completely consistent (no extra noise added)
  # This enhances "high-quality source domain" signal, allowing TransCox to benefit from this part after correcting errors
   other_indices <- setdiff(1:p, c(flip_indices, false_active_indices))
   beta_aux[other_indices] <- beta_aux[other_indices]


  # Generate Weibull survival times (referencing low-dimensional formula)
  gen_weibull <- function(X, beta, noise_sd = 0.5, shape = 1.6) {
    XB <- as.vector(X %*% beta)
    # Inject specified strength of risk noise
    eps <- rnorm(length(XB), 0, noise_sd)
    scale_vec <- exp(-(XB + eps) / 2)
    T <- stats::rweibull(n = length(XB), shape = shape, scale = scale_vec)
    # Approximate control of censoring rate: use quantile as upper bound
    c_upper <- stats::quantile(T, probs = min(0.85, 1 - censoring_rate))
    C <- stats::runif(length(XB), 0, as.numeric(c_upper))
    time   <- ifelse(T < C, T, C)
    status <- ifelse(T < C, 2, 1)  # 2=Event, 1=Censored (Consistent with GenSimData)

    # Build data frame and fix column names
    df <- data.frame(time = time, status = status, X)
    colnames(df)[3:ncol(df)] <- colnames(X)
    return(df)
  }

  # Generate data for each domain
  # Primary domain: High noise (noise_sd = 1.2), simulates low-quality data
  main_data <- gen_weibull(X_main, beta_true, noise_sd = 1.2)

  # Auxiliary domain: Low noise (noise_sd = 0.2), simulates high-quality historical data
  # TransCox can significantly improve performance by transferring high-quality shared knowledge
  aux_data  <- gen_weibull(X_aux, beta_aux, noise_sd = 0.2)

  # Test domain: Medium noise (noise_sd = 0.5)
  test_data <- gen_weibull(X_test, beta_true, noise_sd = 0.5)

  if (verbose) {
    cat("Data generation completed!\n")
    cat(sprintf("Primary domain event rate: %.1f%%\n", mean(main_data$status == 2) * 100))
    cat(sprintf("Source domain event rate: %.1f%%\n", mean(aux_data$status == 2) * 100))
    cat(sprintf("Test event rate: %.1f%%\n", mean(test_data$status == 2) * 100))
  }

  list(
    main_data = main_data,
    aux_data = aux_data,
    test_data = test_data,
    beta_true = beta_true,
    active_features = active_indices,
    data_info = list(
      n_main = n_main,
      n_aux = n_aux,
      n_test = n_test,
      p = p,
      p_active = p_active,
      transfer_strength = transfer_strength,
      rho = rho
    )
  )
}
