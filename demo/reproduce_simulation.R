# =============================================================================
# TransCox-Sparse vs Lasso (Prim/Comb)
# Monte Carlo Simulation (100 Runs)
# =============================================================================
# Sys.setenv(RETICULATE_PYTHON = "change to your python dir")
# --- 1. Environment Setup ---
rm(list = ls())
gc()

# === Configuration ===
save_dir <- file.path(tempdir(), "TransCox_Reproduce_Simulations")
if(!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)

N_SIMULATIONS <- 48
FIXED_SEED_START <- 123
PYTHON_PATH <- "D:/anaconda3/envs/TransCoxEnvi/python.exe" # Path for workers !! change to your worker dir

# Load libraries needed for the main process
library(TransCoxSparse)
library(future.apply) # The key to stable parallelism
library(dplyr)
library(ggplot2)
library(reshape2)
library(knitr)
library(survival)
library(glmnet)
library(TransCox)
library(timeROC)
library(reticulate)

# --- 2. Define the Simulation Worker Function ---
# This function encapsulates ONE entire simulation run.
# It will be sent to different CPU cores to run independently.
run_single_simulation <- function(i, seed_start, py_path) {

    # A. Setup Environment for THIS Worker
    # Each worker needs its own libraries and Python connection

    # Configure Python for this worker
    Sys.setenv(HDF5_DISABLE_VERSION_CHECK = "1")
    Sys.setenv(RETICULATE_PYTHON = py_path)
    use_python(Sys.getenv("RETICULATE_PYTHON"), required = TRUE)

    # Load Source Files (Robustly)
    r_files <- list.files(file.path(getwd(), "R"), pattern = "\\.[rR]$", full.names = TRUE)
    invisible(lapply(r_files, source))

    reticulate::source_python(file.path(getwd(), "inst", "python", "TransCoxFunction_Sparse.py"))

    # B. Set Seed
    current_seed <- seed_start + i
    set.seed(current_seed)
    py_set_seed(current_seed)

    # --- C. Simulation Logic (Copy of your original logic) ---

    # 1. Data Generation
    n_prim <- 100; n_aux <- 500; n_test <- 300; p <- 200; n_active <- 20
    active_indices <- sort(sample(1:p, n_active))
    true_beta <- rep(0, p)
    true_beta[active_indices] <- rnorm(n_active, mean = 0, sd = 0.5)

    sparse_data <- generate_sparse_survival_data(
        n_main = n_prim, n_aux = n_aux, n_test = n_test,
        p = p, p_active = n_active,
        beta_true = true_beta,
        transfer_strength = 0.9, noise_level = 0.5,
        censoring_rate = 0.4, seed = current_seed, verbose = FALSE
    )

    prim_data <- sparse_data$main_data
    aux_data  <- sparse_data$aux_data
    test_data <- sparse_data$test_data
    true_beta_vec <- as.vector(sparse_data$beta_true)
    feature_names <- paste0("X", 1:p)

    # 2. Helper: Evaluation Function (Defined inside worker to ensure availability)
    calculate_ibs_custom <- function(beta_est, train_data, test_data, time_points = 100) {
        tryCatch({
            X_train <- as.matrix(train_data[, feature_names])
            if(sum(abs(beta_est)) < 1e-9) return(0.25)

            risk_train <- as.vector(X_train %*% beta_est)
            df_train <- data.frame(time = train_data$time, status = (train_data$status == 2), lp = risk_train)
            dummy_fit <- coxph(Surv(time, status) ~ offset(lp), data = df_train)
            base_surv <- survfit(dummy_fit, newdata = data.frame(lp = 0))

            X_test_local <- as.matrix(test_data[, feature_names])
            risk_test <- as.vector(X_test_local %*% beta_est)
            eval_times <- seq(min(test_data$time), max(test_data$time), length.out = time_points)
            s0_function <- stepfun(base_surv$time, c(1, base_surv$surv))
            s0_at_eval <- s0_function(eval_times)
            surv_probs <- outer(exp(risk_test), s0_at_eval, function(r, s) s^r)

            bs_list <- numeric(length(eval_times))
            for(j in 1:length(eval_times)) {
                t_val <- eval_times[j]
                known_mask <- (test_data$time > t_val) | (test_data$status == 2)
                if(sum(known_mask) == 0) next
                true_status <- ifelse(test_data$time > t_val, 1, 0)
                bs_list[j] <- mean((true_status[known_mask] - surv_probs[known_mask, j])^2)
            }
            diff_times <- diff(eval_times)
            avg_bs <- (bs_list[1:(length(bs_list)-1)] + bs_list[2:length(bs_list)]) / 2
            ibs <- sum(avg_bs * diff_times) / (max(eval_times) - min(eval_times))
            return(ibs)
        }, error = function(e) return(NA))
    }

    eval_performance_extended <- function(coef_est, name, train_dat_full) {
        X_test <- as.matrix(test_data[, feature_names])
        y_test <- survival::Surv(test_data$time, test_data$status == 2)
        risk_score <- as.vector(X_test %*% coef_est)

        c_index <- survival::concordance(y_test ~ risk_score, reverse = TRUE)$concordance

        median_time <- median(test_data$time[test_data$status == 2])
        auc_val <- tryCatch({
            roc_res <- timeROC::timeROC(
                T = test_data$time, delta = ifelse(test_data$status == 2, 1, 0),
                marker = risk_score, cause = 1, weighting = "marginal",
                times = median_time, iid = FALSE
            )
            roc_res$AUC[[2]]
        }, error = function(e) NA)

        ibs_val <- calculate_ibs_custom(coef_est, train_dat_full, test_data)

        true_active <- abs(true_beta_vec) > 1e-6
        est_active  <- abs(coef_est) > 1e-6
        tp <- sum(true_active & est_active); fp <- sum(est_active & !true_active); fn <- sum(true_active & !est_active)
        precision <- ifelse((tp + fp) == 0, 0, tp / (tp + fp))
        recall    <- ifelse((tp + fn) == 0, 0, tp / (tp + fn))
        f1        <- ifelse((precision + recall) == 0, 0, 2 * precision * recall / (precision + recall))
        mse       <- mean((coef_est - true_beta_vec)^2)

        return(list(
            Method = name, C_index = c_index, AUC_Median = auc_val, IBS = ibs_val,
            MSE = mse, F1 = f1, Precision = precision, Recall = recall, Selected_Num = sum(est_active)
        ))
    }

    # 3. Model Training
    # A. Lasso Primary
    lasso_fit <- train_cox_lasso(train_data = prim_data, cov_names = feature_names, alpha = 1, verbose = FALSE)
    lasso_coef <- as.vector(lasso_fit$coefficients)
    res_lasso_prim <- eval_performance_extended(lasso_coef, "Lasso (Primary)", prim_data)

    # B. Lasso Combined
    combined_data <- rbind(prim_data, aux_data)
    lasso_comb_fit <- train_cox_lasso(train_data = combined_data, cov_names = feature_names, alpha = 1, verbose = FALSE)
    lasso_comb_coef <- as.vector(lasso_comb_fit$coefficients)
    res_lasso_comb <- eval_performance_extended(lasso_comb_coef, "Lasso (Combined)", combined_data)

    # C. TransCox
    # Note: Ensure parallel is FALSE here inside the worker to avoid nested parallel conflicts
    transcox_result <- runTransCox_Sparse(
        primData = prim_data, auxData = aux_data, cov = feature_names, statusvar = "status",
        lambda1 = NULL, lambda2 = NULL,lambda_beta = seq(0.0090, 0.0120, by = 0.0002),
        auto_tune = TRUE, verbose = FALSE, parallel = FALSE
    )
    trans_coef <- as.vector(transcox_result$new_beta)
    res_transcox <- eval_performance_extended(trans_coef, "TransCox-Sparse", prim_data)

    # 4. Return Data
    res_df <- rbind(as.data.frame(res_lasso_prim), as.data.frame(res_lasso_comb), as.data.frame(res_transcox))
    res_df$Simulation_ID <- i

    coef_df <- data.frame(
        Simulation_ID = i, Feature_Index = 1:p, Is_True_Active = (abs(true_beta_vec) > 1e-6),
        Beta_Lasso_Prim = lasso_coef, Beta_Lasso_Comb = lasso_comb_coef, Beta_TransCox = trans_coef
    )

    return(list(metrics = res_df, coefs = coef_df))
}

# --- 3. Run Parallel Simulations ---

# Setup Parallel Plan
# Use "multisession" for Windows (creates background R processes)
n_cores <- parallel::detectCores(logical = FALSE) - 1
n_workers <- min(n_cores, 20) # Limit workers to save RAM (Python processes are heavy)

cat(sprintf(">> Configuring Parallel Backend with %d workers...\n", n_workers))
plan(multisession, workers = n_workers)

cat(">> Starting Simulations via future_lapply...\n")
time_start <- Sys.time()

# Run Simulations
# future.seed = TRUE automatically handles random number generation safely
results_list <- future_lapply(1:N_SIMULATIONS, function(i) {
    run_single_simulation(i, FIXED_SEED_START, PYTHON_PATH)
}, future.seed = TRUE)

time_end <- Sys.time()
cat(sprintf(">> Simulation Completed in %.2f minutes.\n", as.numeric(diff(c(time_start, time_end), units="mins"))))

# --- 4. Process & Visualize Results (Aggregating back on main process) ---

cat(">> Processing Results...\n")

# Unpack lists
all_metrics <- lapply(results_list, function(x) x$metrics)
all_coefficients <- lapply(results_list, function(x) x$coefs)

df_metrics <- do.call(rbind, all_metrics)
df_coefs <- do.call(rbind, all_coefficients)

# 4.1 Summary Stats
summary_table <- df_metrics %>%
    group_by(Method) %>%
    summarise(
        C_index_Mean = mean(C_index, na.rm=T), C_index_SD = sd(C_index, na.rm=T),
        IBS_Mean = mean(IBS, na.rm=T), IBS_SD = sd(IBS, na.rm=T),
        AUC_Mean = mean(AUC_Median, na.rm=T), AUC_SD = sd(AUC_Median, na.rm=T),
        Selected_Mean = mean(Selected_Num, na.rm=T)
    )

print(kable(summary_table, digits = 4, caption = "Pooled Performance"))
write.csv(summary_table, file.path(save_dir, "Summary_Statistics_Mean_SD.csv"), row.names = FALSE)
write.csv(df_metrics, file.path(save_dir, "Raw_Metrics_100Runs.csv"), row.names = FALSE)

# 4.2 Visualization
p_box_cindex <- ggplot(df_metrics, aes(x = Method, y = C_index, fill = Method)) +
    geom_boxplot(alpha = 0.7, outlier.shape = NA) +
    geom_jitter(width = 0.2, alpha = 0.3, size = 1) +
    labs(title = "C-index Distribution (100 Runs)", y = "C-index") +
    theme_bw() + theme(legend.position = "none") +
    scale_fill_manual(values = c("#E69F00", "#56B4E9", "#D55E00"))

ggsave(file.path(save_dir, "Boxplot_Cindex.pdf"), p_box_cindex, width = 6, height = 5)

# Selection Probability
df_coefs_long <- df_coefs %>%
    select(Simulation_ID, Feature_Index, Is_True_Active, Beta_Lasso_Prim, Beta_Lasso_Comb, Beta_TransCox) %>%
    melt(id.vars = c("Simulation_ID", "Feature_Index", "Is_True_Active"), variable.name = "Method", value.name = "Beta_Value") %>%
    mutate(Selected = abs(Beta_Value) > 1e-6) %>%
    mutate(Method = gsub("Beta_", "", Method))

plot_prob_data <- df_coefs_long %>%
    group_by(Method, Is_True_Active) %>%
    summarise(Selection_Prob = mean(Selected), .groups = 'drop') %>%
    mutate(Variable_Type = ifelse(Is_True_Active, "True Active", "Noise"))

p_sel <- ggplot(plot_prob_data, aes(x = Method, y = Selection_Prob, fill = Method)) +
    geom_bar(stat = "identity", position = position_dodge(), width = 0.7) +
    facet_wrap(~Variable_Type, scales = "free_y") +
    labs(title = "Selection Probability", y = "Probability") +
    theme_bw() + scale_fill_manual(values = c("#E69F00", "#56B4E9", "#D55E00"))

ggsave(file.path(save_dir, "Selection_Probability.pdf"), p_sel, width = 8, height = 5)

cat("\nDone! Results saved to:", save_dir, "\n")
