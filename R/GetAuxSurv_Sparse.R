#' Source Domain Parameter Estimation for High-Dimensional Sparse Data
#'
#' @description
#' Estimates source domain parameters using Lasso-Cox regression, supporting high-dimensional sparse data.
#'
#' @param auxData Source domain data.
#' @param cov Vector of covariate names.
#' @param lambda_aux Lasso penalty parameter. If NULL, selected automatically.
#' @param alpha Elastic net parameter. 1 for Lasso, 0 for Ridge.
#' @param cv_folds Number of cross-validation folds.
#' @param weights Sample weights.
#'
#' @return A list containing estR and q.
#'
#' @importFrom stats as.formula coef
#' @importFrom survival Surv coxph basehaz
#' @importFrom glmnet cv.glmnet glmnet
#' @export
GetAuxSurv_Sparse <- function(auxData, cov = c("X1", "X2"),
                              lambda_aux = NULL, alpha = 1,
                              cv_folds = 5, weights = NULL) {

  # Estimate source domain parameters using Lasso-Cox regression

  # Check data dimensions
  n_samples <- nrow(auxData)
  n_features <- length(cov)

  # Prepare data
  X_matrix <- as.matrix(auxData[, cov])
  time_var <- auxData$time
  status_var <- auxData$status

  # Check data quality - support 0/1 and 1/2 encoding
  if (all(status_var %in% c(0, 1))) {
    # 0/1 encoding
    event_rate <- mean(status_var == 1)
    surv_obj <- survival::Surv(time_var, status_var)
    status_target <- 1
  } else if (all(status_var %in% c(1, 2))) {
    # 1/2 encoding
    event_rate <- mean(status_var == 2)
    surv_obj <- survival::Surv(time_var, status_var == 2)
    status_target <- 2
  } else {
    stop("Unsupported status encoding. Please use 0/1 or 1/2 encoding.")
  }

  # Fallback function for standard Cox
  run_standard_cox <- function(data, cov_vars, wts) {
    formula_str <- paste("survival::Surv(time, status == ", status_target, ") ~ ", paste(cov_vars, collapse = "+"), sep="")
    res.cox <- survival::coxph(stats::as.formula(formula_str), data = data, weights = wts)
    bhest <- survival::basehaz(res.cox, centered=FALSE)
    estR <- res.cox$coefficients
    q <- data.frame(cumHazards = bhest$hazard,
                    breakPoints = bhest$time)
    return(list(estR = estR, q = q))
  }

  if (event_rate < 0.05) {
    warning("Event rate too low (", round(event_rate, 3), "), falling back to standard Cox regression.")
    return(run_standard_cox(auxData, cov, weights))
  }

  # Set weights
  if (is.null(weights)) {
    weights <- rep(1, n_samples)
  }

  # Try using glmnet
  tryCatch({
    if (is.null(lambda_aux)) {
      # Select optimal lambda using cross-validation

      # Set lambda sequence
      lambda_seq <- exp(seq(log(0.001), log(1), length.out = 50))

      # Cross-validation
      cv_fit <- glmnet::cv.glmnet(
        x = X_matrix,
        y = surv_obj,
        family = "cox",
        alpha = alpha,
        lambda = lambda_seq,
        nfolds = min(cv_folds, n_samples),
        type.measure = "deviance",
        weights = weights
      )

      lambda_optimal <- cv_fit$lambda.1se
    } else {
      lambda_optimal <- lambda_aux
    }

    # Fit final model
    lasso_fit <- glmnet::glmnet(
      x = X_matrix,
      y = surv_obj,
      family = "cox",
      alpha = alpha,
      lambda = lambda_optimal,
      weights = weights
    )

    # Extract coefficients
    estR_sparse <- as.vector(stats::coef(lasso_fit, s = lambda_optimal))
    names(estR_sparse) <- cov

    # Calculate baseline cumulative hazards
    # Use standard Cox model to calculate baseline hazards
    if (sum(abs(estR_sparse) > 1e-8) > 0) {
      # If there are non-zero coefficients, use these coefficients
      nonzero_idx <- which(abs(estR_sparse) > 1e-8)
      if (length(nonzero_idx) > 0 && length(nonzero_idx) < n_samples - 5) {
        X_nonzero <- X_matrix[, nonzero_idx, drop = FALSE]
        cov_nonzero <- cov[nonzero_idx]

        cox_data <- data.frame(
          time = time_var,
          status = as.numeric(status_var == status_target),
          X_nonzero
        )
        colnames(cox_data)[3:ncol(cox_data)] <- cov_nonzero

        formula_str <- paste("survival::Surv(time, status) ~", paste(cov_nonzero, collapse = " + "))
        cox_formula <- stats::as.formula(formula_str)

        cox_fit <- survival::coxph(cox_formula, data = cox_data, weights = weights)
        bhest <- survival::basehaz(cox_fit, centered = FALSE)
      } else {
        # If too many non-zero coefficients, use intercept-only model
        cox_fit <- survival::coxph(survival::Surv(time_var, status_var == status_target) ~ 1, weights = weights)
        bhest <- survival::basehaz(cox_fit, centered = FALSE)
      }
    } else {
      # If all coefficients are zero, use intercept-only model
      cox_fit <- survival::coxph(survival::Surv(time_var, status_var == status_target) ~ 1, weights = weights)
      bhest <- survival::basehaz(cox_fit, centered = FALSE)
    }

    q <- data.frame(
      cumHazards = bhest$hazard,
      breakPoints = bhest$time
    )

    # Calculate sparsity statistics
    nonzero_count <- sum(abs(estR_sparse) > 1e-8)
    sparsity_ratio <- 1 - nonzero_count / length(estR_sparse)

    # Lasso-Cox fit successful

    return(list(
      estR = estR_sparse,
      q = q,
      lambda_used = lambda_optimal,
      nonzero_count = nonzero_count,
      total_features = length(estR_sparse),
      sparsity_ratio = sparsity_ratio,
      method = "lasso_cox"
    ))

  }, error = function(e) {
    # Lasso-Cox fit failed, fallback to standard Cox regression
    warning("Lasso-Cox fit failed, falling back to standard Cox regression: ", e$message)
    return(run_standard_cox(auxData, cov, weights))
  })
}
