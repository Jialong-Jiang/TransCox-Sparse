# =============================================================================
# TransCox-Sparse vs Lasso (Prim/Comb)
# Includes: Time-dependent AUC, IBS (Breslow Est), KM Curves, Coefficient Export
# =============================================================================
# Sys.setenv(RETICULATE_PYTHON = "change to your python dir")


# --- 1. Environment Setup ---
rm(list = ls())
gc()


# Load libraries needed for the main process
# library(TransCoxSparse)
# library(dplyr)
# library(ggplot2)
# library(reshape2)
# library(knitr)
# library(survival)
# library(glmnet)
# library(TransCox)
# library(timeROC)
# library(reticulate)
# library(survminer)
# library(survival)

# === Configuration ===
save_dir <- file.path(tempdir(), "TransCox_Quick_Start_Results") #change to your dir
if(!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)

PYTHON_PATH <- "D:/anaconda3/envs/TransCoxEnvi/python.exe" # Path for workers #change to your worker dir
Sys.setenv(HDF5_DISABLE_VERSION_CHECK = "1")
Sys.setenv(RETICULATE_PYTHON = PYTHON_PATH)
use_python(Sys.getenv("RETICULATE_PYTHON"), required = TRUE)


reticulate::source_python(file.path(getwd(), "inst", "python", "TransCoxFunction_Sparse.py"))


my_seed <- 123 # for 100 times experiments and more details, please see in 100.R
# 1. Fix R random seed
set.seed(my_seed)

# 2. Fix Python random seed (via reticulate)
# This is important for TransCox as it has random initialization in Python
reticulate::py_set_seed(my_seed)

cat("=== Simulation Start: Lasso(Prim) vs Lasso(Comb) vs TransCox(Auto) ===\n")

# --- 2. Data Generation ---
cat(">> Step 1: Generating Data\n")
n_prim <- 100; n_aux <- 500; n_test <- 300; p <- 200; n_active <- 20
active_indices <- sort(sample(1:p, n_active))
true_beta <- rep(0, p)
#true_beta[active_indices] <- rnorm(n_active, mean = 0.5, sd = 0.2)
true_beta[active_indices] <- rnorm(n_active, mean = 0, sd = 0.5)
sparse_data <- generate_sparse_survival_data(
    n_main = n_prim, n_aux = n_aux, n_test = n_test,
    p = p, p_active = n_active,
    beta_true = true_beta,
    transfer_strength = 0.9, noise_level = 0.5,
    censoring_rate = 0.4, seed = my_seed, verbose = FALSE
)

prim_data <- sparse_data$main_data
aux_data  <- sparse_data$aux_data
test_data <- sparse_data$test_data
true_beta <- as.vector(sparse_data$beta_true)
feature_names <- paste0("X", 1:p)

X_test <- as.matrix(test_data[, feature_names])
y_test <- survival::Surv(test_data$time, test_data$status == 2)


# --- Core Helper Function: IBS Calculation based on Breslow Estimator ---
# Explanation: Cox model only provides beta. To calculate IBS, baseline hazard S0(t) must be estimated.
# This function uses the training set to estimate S0(t), combines it with the test set Risk Score to calculate predicted survival probability S(t|x),
# and finally calculates the Integrated Brier Score.
calculate_ibs_custom <- function(beta_est, train_data, test_data, time_points = 100) {
    tryCatch({
        # 1. Prepare training data
        X_train <- as.matrix(train_data[, feature_names])
        y_train <- survival::Surv(train_data$time, train_data$status == 2)

        # 2. If Beta is all 0 (model failure), return IBS of null model (based on Kaplan-Meier)
        if(sum(abs(beta_est)) < 1e-9) {
            # Simplified handling: return NA or a large value
            return(0.25) # 0.25 is the expected BS of random guess
        }

        # 3. Calculate Risk Score for training set
        risk_train <- as.vector(X_train %*% beta_est)

        # 4. Build a fixed coefficient Cox model object (iter=0) to get baseline hazard
        # This is a common trick: coxph accepts init parameter
        # We build a data frame containing linear predictor
        df_train <- data.frame(time = train_data$time, status = (train_data$status == 2), lp = risk_train)
        # Fit offset model to estimate baseline hazard
        dummy_fit <- coxph(Surv(time, status) ~ offset(lp), data = df_train)

        # 5. Get baseline survival function S0(t)
        # survfit returns baseline survival rate (i.e., survival rate when lp=0)
        base_surv <- survfit(dummy_fit, newdata = data.frame(lp = 0))

        # 6. Predict survival probability matrix for test set
        X_test_local <- as.matrix(test_data[, feature_names])
        risk_test <- as.vector(X_test_local %*% beta_est)

        # Define evaluation time points (from test set min to max, 100 points)
        eval_times <- seq(min(test_data$time), max(test_data$time), length.out = time_points)

        # Interpolate to get S0(t) at these time points
        # summary(base_surv) may not contain all time points, need step function interpolation
        s0_function <- stepfun(base_surv$time, c(1, base_surv$surv))
        s0_at_eval <- s0_function(eval_times)

        # S(t|x) = S0(t)^exp(lp)
        surv_probs <- outer(exp(risk_test), s0_at_eval, function(r, s) s^r)
        # Result matrix: row=sample, col=time point

        # 7. Calculate Brier Score (Simplified, without IPCW weights, to avoid dependency errors)
        # Standard Brier Score: BS(t) = 1/N * sum( (I(T>t) - S(t|x))^2 )
        # Note: Strictly speaking, IPCW (Inverse Probability Censoring Weighting) is needed
        # For code robustness, use unweighted version here, or call SurvMetrics if installed

        bs_list <- numeric(length(eval_times))
        for(j in 1:length(eval_times)) {
            t_val <- eval_times[j]
            # True status: 1=alive past t, 0=dead before t.
            # Handling censoring: If censored before t, status is unknown, usually need IPCW.
            # Simple handling: Only calculate for samples with known status (Robust simple implementation)
            known_mask <- (test_data$time > t_val) | (test_data$status == 2)
            if(sum(known_mask) == 0) next

            true_status <- ifelse(test_data$time > t_val, 1, 0)
            pred_surv <- surv_probs[, j]

            # Calculate MSE only on known samples
            bs_list[j] <- mean((true_status[known_mask] - pred_surv[known_mask])^2)
        }

        # 8. Integrate to get IBS (using trapezoidal rule)
        diff_times <- diff(eval_times)
        avg_bs <- (bs_list[1:(length(bs_list)-1)] + bs_list[2:length(bs_list)]) / 2
        ibs <- sum(avg_bs * diff_times) / (max(eval_times) - min(eval_times))

        return(ibs)

    }, error = function(e) {
        warning("IBS calculation failed: ", e$message)
        return(NA)
    })
}

# --- Evaluation Function (Integrates C-index, AUC, IBS, MSE) ---
eval_performance_extended <- function(coef_est, name, train_dat_full) {
    risk_score <- as.vector(X_test %*% coef_est)

    # 1. C-index
    c_index <- survival::concordance(y_test ~ risk_score, reverse = TRUE)$concordance

    # 2. AUC (Median Time)
    median_time <- median(test_data$time[test_data$status == 2])
    auc_val <- tryCatch({
        roc_res <- timeROC::timeROC(
            T = test_data$time, delta = ifelse(test_data$status == 2, 1, 0),
            marker = risk_score, cause = 1, weighting = "marginal",
            times = median_time, iid = FALSE
        )
        roc_res$AUC[[2]]
    }, error = function(e) NA)

    # 3. IBS (New!)
    ibs_val <- calculate_ibs_custom(coef_est, train_dat_full, test_data)

    # 4. Coefficients Metrics
    true_active <- abs(true_beta) > 1e-6
    est_active  <- abs(coef_est) > 1e-6
    tp <- sum(true_active & est_active)
    fp <- sum(est_active & !true_active)
    fn <- sum(true_active & !est_active)

    precision <- ifelse((tp + fp) == 0, 0, tp / (tp + fp))
    recall    <- ifelse((tp + fn) == 0, 0, tp / (tp + fn))
    f1        <- ifelse((precision + recall) == 0, 0, 2 * precision * recall / (precision + recall))
    mse       <- mean((coef_est - true_beta)^2)

    return(list(
        Method = name,
        C_index = c_index,
        AUC_Median = auc_val,
        IBS = ibs_val,        # New output
        MSE = mse,
        F1 = f1,
        Selected = sum(est_active)
    ))
}

# --- Plotting Function ---
plot_save_km <- function(coef_est, model_name, data_test, save_path) {
    X_mat <- as.matrix(data_test[, feature_names])
    risk_scores <- as.vector(X_mat %*% coef_est)
    group <- ifelse(risk_scores > median(risk_scores), "High Risk", "Low Risk")
    surv_df <- data.frame(time = data_test$time, status = data_test$status == 2,
                          group = factor(group, levels = c("Low Risk", "High Risk")))
    fit <- survfit(Surv(time, status) ~ group, data = surv_df)

    p <- ggsurvplot(
        fit, data = surv_df, pval = TRUE, conf.int = TRUE, risk.table = TRUE,
        palette = c("#2E9FDF", "#E7B800"),
        title = paste0("KM Curve: ", model_name),
        xlab = "Time", ylab = "Survival Probability", ggtheme = theme_bw()
    )

    pdf_file <- paste0(save_path, "KM_", gsub(" ", "_", model_name), ".pdf")
    pdf(pdf_file, width = 7, height = 7, onefile = FALSE)
    print(p)
    dev.off()
    cat(sprintf("   -> KM Curve saved: %s\n", pdf_file))
}

# --- 3. Model Training and Evaluation ---

# A. Lasso (Primary)

cat(">> Training Lasso (Primary)...\n")
lasso_fit <- train_cox_lasso(train_data = prim_data, cov_names = feature_names, alpha = 1, verbose = FALSE)
lasso_coef <- as.vector(lasso_fit$coefficients)
# Note: Passing training set here for IBS estimation
res_lasso_prim <- eval_performance_extended(lasso_coef, "Lasso (Primary)", prim_data)

# B. Lasso (Combined)
cat(">> Training Lasso (Combined)...\n")

combined_data <- rbind(prim_data, aux_data)
lasso_comb_fit <- train_cox_lasso(train_data = combined_data, cov_names = feature_names, alpha = 1, verbose = FALSE)
lasso_comb_coef <- as.vector(lasso_comb_fit$coefficients)
res_lasso_comb <- eval_performance_extended(lasso_comb_coef, "Lasso (Combined)", combined_data)

# C. TransCox (Auto)
cat(">> Training TransCox-Sparse (Auto)...\n")
transcox_start_time <- Sys.time()


transcox_result <- runTransCox_Sparse(
    primData = prim_data, auxData = aux_data, cov = feature_names, statusvar = "status",
    lambda1 = NULL, lambda2 = NULL, lambda_beta = seq(0.009, 0.010, by = 0.0002),
    auto_tune = TRUE, verbose = FALSE
)
trans_coef <- as.vector(transcox_result$new_beta)
# TransCox optimizes based on Primary objective, so use Primary data for baseline hazard
res_transcox <- eval_performance_extended(trans_coef, "TransCox-Sparse", prim_data)


# --- 4. Result Aggregation and Export ---
cat("\n>> Step 5: Aggregating and Exporting Results\n")

# 4.1 Summary Metrics Table
results_df <- rbind(
    as.data.frame(res_lasso_prim),
    as.data.frame(res_lasso_comb),
    as.data.frame(res_transcox)
)

# Print to console
print(knitr::kable(results_df, digits = 4, caption = "Final Model Performance Comparison (including IBS)"))

# Save metrics to CSV

metrics_file <- paste0(save_dir, "metrics_summary.csv")
write.csv(results_df, file = metrics_file, row.names = FALSE)
cat(sprintf("   -> Performance metrics saved to: %s\n", metrics_file))

# 4.2 Coefficient Table (Output first 50 features as example, but save all)
coef_df <- data.frame(
    Feature = feature_names,
    True_Beta = true_beta,
    Lasso_Prim = lasso_coef,
    Lasso_Comb = lasso_comb_coef,
    TransCox = trans_coef
)
coef_file <- paste0(save_dir, "coefficients_all.csv")
write.csv(coef_df, file = coef_file, row.names = FALSE)
cat(sprintf("   -> Full coefficient table saved to: %s\n", coef_file))

# --- 5. Visualization ---
cat("\n>> Step 6: Generating Visualizations\n")

get_km_plot_object <- function(coef_est, model_name, data_test) {

    X_mat <- as.matrix(data_test[, feature_names])
    risk_scores <- as.vector(X_mat %*% coef_est)

    group <- ifelse(risk_scores > median(risk_scores), "High Risk", "Low Risk")
    surv_df <- data.frame(time = data_test$time, status = data_test$status == 2,
                          group = factor(group, levels = c("Low Risk", "High Risk")))

    fit <- survfit(Surv(time, status) ~ group, data = surv_df)

    p <- ggsurvplot(
        fit, data = surv_df,
        pval = TRUE,
        pval.size = 4,
        conf.int = TRUE,
        risk.table = TRUE,
        risk.table.height = 0.25,
        palette = c("#2E9FDF", "#E7B800"),
        title = model_name,
        xlab = "Time", ylab = "Survival Probability",
        ggtheme = theme_bw(),
        legend.title = "Risk Group",
        legend.labs = c("Low", "High")
    )
    return(p)
}

cat("Generating individual plot objects...\n")
p1 <- get_km_plot_object(lasso_coef, "Lasso (Primary)", test_data)
p2 <- get_km_plot_object(lasso_comb_coef, "Lasso (Combined)", test_data)
p3 <- get_km_plot_object(trans_coef, "TransCox-Sparse", test_data)

cat("Arranging and saving merged plot...\n")

merged_pdf_file <- paste0(save_dir, "Merged_KM_Curves.pdf")

pdf(merged_pdf_file, width = 18, height = 7, onefile = FALSE)

arrange_ggsurvplots(
    list(p1, p2, p3),
    print = TRUE,
    ncol = 3, nrow = 1,
    risk.table.height = 0.25
)

dev.off()

cat(sprintf("   -> Merged KM plot saved to: %s\n", merged_pdf_file))

# Lollipop Plot
cat("Generating coefficient comparison plot...\n")
plot_indices <- active_indices
plot_data <- data.frame(
    Index = rep(plot_indices, 4),
    Value = c(true_beta[plot_indices],
              lasso_coef[plot_indices],
              lasso_comb_coef[plot_indices],
              trans_coef[plot_indices]),
    Method = factor(rep(c("True Beta", "Lasso (Primary)", "Lasso (Combined)", "TransCox"),
                        each = length(plot_indices)),
                    levels = c("True Beta", "Lasso (Primary)", "Lasso (Combined)", "TransCox"))
)

p_coef <- ggplot(plot_data, aes(x = as.factor(Index), y = Value, color = Method, shape = Method)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    geom_point(position = position_dodge(width = 0.6), size = 3, alpha = 0.8) +
    geom_linerange(aes(ymin = 0, ymax = Value), position = position_dodge(width = 0.6), alpha = 0.5) +
    labs(title = "Coefficient Estimation (Active Features)", x = "Feature Index", y = "Coefficient") +
    theme_bw() + theme(legend.position = "bottom") +
    scale_color_manual(values = c("black", "#E69F00", "#56B4E9", "#D55E00"))

ggsave(filename = paste0(save_dir, "Coefficient_Comparison.pdf"), plot = p_coef, width = 12, height = 6)

