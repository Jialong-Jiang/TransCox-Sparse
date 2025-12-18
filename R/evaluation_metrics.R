#' Survival Analysis Evaluation Metrics
#' 
#' @description 
#' Implements common evaluation metrics for survival analysis, including C-index, AUC, Brier Score, etc.
#' 

#' Calculate C-index (Concordance Index)
#'
#' @description
#' Computes the concordance index (C-index) for survival models, which measures
#' the proportion of concordant pairs among all comparable pairs of observations.
#'
#' @param predicted_risk Numeric vector of predicted risk scores. Higher values
#'   indicate higher risk of event occurrence.
#' @param time Numeric vector of survival times.
#' @param status Numeric vector of event indicators (1 = event occurred, 0 = censored).
#' 
#' @return Numeric value representing the C-index (range: 0 to 1, where 0.5 indicates
#'   random prediction and 1.0 indicates perfect prediction).
#' 
#' @examples
#' \dontrun{
#' # Example with simulated data
#' time <- c(1, 2, 3, 4, 5)
#' status <- c(1, 1, 0, 1, 0)
#' risk <- c(0.8, 0.6, 0.4, 0.7, 0.3)
#' cindex <- calculate_cindex(risk, time, status)
#' }
#' 
#' @export
calculate_cindex <- function(predicted_risk, time, status) {
    
    if (length(predicted_risk) != length(time) || length(time) != length(status)) {
        stop("Input vectors must have consistent lengths")
    }
    
    # Ensure 'survival' package is installed
    if (!requireNamespace("survival", quietly = TRUE)) {
        stop("Please install 'survival' package (install.packages('survival'))")
    }
    
    # Create Surv object
    y <- survival::Surv(time, status)
    
    # Calculate C-index using survival::concordance
    # The formula y ~ predicted_risk assumes higher "predicted_risk" means higher risk
    out <- survival::concordance(y ~ predicted_risk)
    
    # Extract C-index value
    return(as.numeric(out$concordance))
}

#' Calculate Time-Dependent AUC
#' 
#' @description
#' Computes the time-dependent Area Under the Curve (AUC) for survival models
#' at a specific time point using the timeROC package.
#' 
#' @param predicted_risk Numeric vector of predicted risk scores. Higher values
#'   indicate higher risk of event occurrence.
#' @param time Numeric vector of survival times.
#' @param status Numeric vector of event indicators (1 = event occurred, 0 = censored).
#' @param time_point Numeric. The specific time point at which to evaluate the AUC.
#' 
#' @return Numeric value representing the time-dependent AUC at the specified time point.
#' 
#' @examples
#' \dontrun{
#' # Example with simulated data
#' time <- c(1, 2, 3, 4, 5)
#' status <- c(1, 1, 0, 1, 0)
#' risk <- c(0.8, 0.6, 0.4, 0.7, 0.3)
#' auc <- calculate_time_dependent_auc(risk, time, status, time_point = 3)
#' }
#' 
#' @export
calculate_time_dependent_auc <- function(predicted_risk, time, status, time_point) {
    
    if (length(predicted_risk) != length(time) || length(time) != length(status)) {
        stop("Input vectors must have consistent lengths")
    }
    if (!requireNamespace("timeROC", quietly = TRUE)) {
        stop("Please install 'timeROC' package (install.packages('timeROC'))")
    }
    
    # Calculate using timeROC::timeROC
    roc_obj <- timeROC::timeROC(
        T = time,
        delta = status,
        marker = predicted_risk,   # Higher risk score means higher risk
        cause = 1,
        weighting = "marginal",  # Default "marginal" weighting
        times = time_point,
        ROC = FALSE              # Only need AUC, not full ROC curve
    )
    
    # timeROC might return multiple time points (if time_point is not in observed times)
    # Select the AUC value closest to the requested time_point
    idx <- which.min(abs(roc_obj$times - time_point))
    
    return(as.numeric(roc_obj$AUC[idx]))
}

#' Calculate Brier Score
#' 
#' @param predicted_survival Predicted survival probabilities at time_point (a vector)
#' @param time Survival times
#' @param status Event indicators
#' @param time_point Evaluation time point
#' @return Brier Score value
#' @export
calculate_brier_score <- function(predicted_survival, time, status, time_point) {
    
    if (length(predicted_survival) != length(time) || length(time) != length(status)) {
        stop("Input vectors must have consistent lengths")
    }
    if (!requireNamespace("ipred", quietly = TRUE)) {
        stop("Please install 'ipred' package (install.packages('ipred'))")
    }
    if (!requireNamespace("survival", quietly = TRUE)) {
        stop("Please install 'survival' package")
    }
    
    # Ensure time_point is a single value
    if (length(time_point) > 1) {
        warning("time_point has multiple values, using only the first one")
        time_point <- time_point[1]
    }

    # Create Surv object
    y <- survival::Surv(time, status)
    
    # ipred::sbrier 'pred' argument requires a matrix
    # Rows correspond to samples in y, columns correspond to time points in btime
    # Since we evaluate only one time point, create a single-column matrix
    pred_matrix <- matrix(predicted_survival, ncol = 1)
    
    # Calculate using ipred::sbrier
    # btime argument specifies the time point to evaluate
    bs <- ipred::sbrier(obj = y, pred = pred_matrix, btime = time_point)
    
    # sbrier returns a vector corresponding to each time point in btime
    # Since we input only one time point, result is bs[1]
    return(as.numeric(bs[1]))
}

#' Calculate Parameter Estimation Accuracy Metrics
#' 
#' @param estimated_beta Estimated regression coefficients
#' @param true_beta True regression coefficients
#' @return A list containing multiple accuracy metrics
#' @export
calculate_parameter_accuracy <- function(estimated_beta, true_beta) {
    
    if (length(estimated_beta) != length(true_beta)) {
        stop("Estimated coefficients and true coefficients have different lengths")
    }
    
    # Mean Squared Error (MSE)
    mse <- mean((estimated_beta - true_beta)^2)
    
    # Root Mean Squared Error (RMSE)
    rmse <- sqrt(mse)
    
    # Mean Absolute Error (MAE)
    mae <- mean(abs(estimated_beta - true_beta))
    
    # Correlation coefficient
    correlation <- cor(estimated_beta, true_beta, use = "complete.obs")
    
    # Sparsity related metrics
    true_active <- which(true_beta != 0)
    estimated_active <- which(abs(estimated_beta) > 1e-6)  # Consider numerical precision
    
    # True Positives
    tp <- length(intersect(true_active, estimated_active))
    
    # False Positives
    fp <- length(setdiff(estimated_active, true_active))
    
    # False Negatives
    fn <- length(setdiff(true_active, estimated_active))
    
    # True Negatives
    tn <- length(true_beta) - tp - fp - fn
    
    # Precision
    precision <- if (tp + fp > 0) tp / (tp + fp) else 0
    
    # Recall (Sensitivity)
    recall <- if (tp + fn > 0) tp / (tp + fn) else 0
    
    # F1 Score
    f1_score <- if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0
    
    # Specificity
    specificity <- if (tn + fp > 0) tn / (tn + fp) else 0
    
    return(list(
        mse = mse,
        rmse = rmse,
        mae = mae,
        correlation = correlation,
        precision = precision,
        recall = recall,
        f1_score = f1_score,
        specificity = specificity,
        true_positives = tp,
        false_positives = fp,
        false_negatives = fn,
        true_negatives = tn,
        n_estimated_active = length(estimated_active),
        n_true_active = length(true_active)
    ))
}

#' Comprehensive Model Performance Evaluation
#' 
#' @param predicted_risk Predicted risk scores
#' @param estimated_beta Estimated regression coefficients
#' @param test_data Test data
#' @param true_beta True regression coefficients
#' @param time_points Time points for evaluation
#' @return A list containing all evaluation metrics
#' @export
comprehensive_evaluation <- function(predicted_risk, estimated_beta, test_data, true_beta, time_points = NULL) {
    
    time <- test_data$time
    status <- test_data$status
    
    # Calculate C-index
    if (length(unique(predicted_risk)) == 1) {
        warning("All predicted risk scores are identical, C-index may be inaccurate")
        return(0.5)
    }
    
    if (sum(status) == 0) {
        warning("No events observed, cannot calculate C-index")
        return(NA)
    }
    cindex <- calculate_cindex(predicted_risk, time, status)
    
    # Parameter accuracy
    param_accuracy <- NULL
    if (!is.null(estimated_beta) && !is.null(true_beta)) {
        # Ensure coefficient lengths match
        if (length(estimated_beta) == length(true_beta)) {
            param_accuracy <- calculate_parameter_accuracy(estimated_beta, true_beta)
        } else {
            # If lengths mismatch, calculate only basic stats
            param_accuracy <- list(
                estimated_length = length(estimated_beta),
                true_length = length(true_beta),
                estimated_nonzero = sum(abs(estimated_beta) > 1e-6),
                true_nonzero = sum(abs(true_beta) > 1e-6),
                note = "Coefficient lengths mismatch, cannot calculate detailed accuracy metrics"
            )
        }
    }
    
    # Time-dependent metrics
    time_dependent_metrics <- NULL
    if (!is.null(time_points)) {
        time_dependent_metrics <- list()
        for (t in time_points) {
            if (t < max(time[status == 1])) {  # Ensure time point is meaningful
                auc_t <- calculate_time_dependent_auc(predicted_risk, time, status, t)
                time_dependent_metrics[[paste0("AUC_", t)]] <- auc_t
            }
        }
    }
    
    # Distribution statistics of predicted risk
    risk_stats <- list(
        mean_risk = mean(predicted_risk),
        sd_risk = sd(predicted_risk),
        min_risk = min(predicted_risk),
        max_risk = max(predicted_risk),
        range_risk = max(predicted_risk) - min(predicted_risk)
    )
    
    # Create summary
    summary_stats <- list(cindex = round(cindex, 4))
    
    if (!is.null(param_accuracy) && !is.null(param_accuracy$correlation)) {
        summary_stats$correlation <- round(param_accuracy$correlation, 4)
        summary_stats$f1_score <- round(param_accuracy$f1_score, 4)
        summary_stats$precision <- round(param_accuracy$precision, 4)
        summary_stats$recall <- round(param_accuracy$recall, 4)
    } else {
        summary_stats$note <- "Coefficient lengths mismatch, cannot calculate parameter accuracy metrics"
    }
    
    return(list(
        cindex = cindex,
        parameter_accuracy = param_accuracy,
        time_dependent_metrics = time_dependent_metrics,
        risk_statistics = risk_stats,
        summary = summary_stats
    ))
}
