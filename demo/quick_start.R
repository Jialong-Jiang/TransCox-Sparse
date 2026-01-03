# =============================================================================
# TransCox-Sparse vs Lasso (Prim/Comb)
# Includes: Time-dependent AUC, IBS (Breslow Est), KM Curves, Coefficient Export
# =============================================================================
# Sys.setenv(RETICULATE_PYTHON = "change to your python dir")


# --- 1. Environment Setup ---
rm(list = ls())
gc()

# Load libraries
library(dplyr)
library(tidyr)          # For pivot_longer
library(ggplot2)
library(knitr)
library(survival)
library(glmnet)
library(timeROC)
library(survminer)
library(pec)
library(TransCoxSparse)

# === Configuration ===
save_dir <- file.path(tempdir(), "TransCox_Quick_Start_Results") #change to your dir
if(!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)

library(reticulate)
use_condaenv("TransCoxEnvi")
source_python(system.file("python", "TransCoxFunction_Sparse.py", package = "TransCoxSparse"))


my_seed <- 123
set.seed(my_seed)
if(exists("py_set_seed")) try(reticulate::py_set_seed(my_seed), silent=TRUE)

# --- 3. Data Generation -----------------
cat(">> Step 1: Generating Data (N=300, P=500)...\n")
p <- 500; n_prim <- 300; n_aux <- 1000; n_test <- 300
feature_names <- paste0("X", 1:p)

# Truth
beta_prim <- rep(0, p); beta_prim[1:10] <- 1.0; beta_prim[11:20] <- -1.0

# Source
beta_aux <- rep(0, p)
beta_aux[1:10] <- 1.0 + rnorm(10, 0, 0.1)  # Group A: Consistent
beta_aux[11:15] <- 1.0                     # Group B: Conflict
beta_aux[16:20] <- 0                       # Group C: Missing
beta_aux[21:25] <- 0.5                     # Group D: Source Noise (21-25)
beta_aux <- beta_aux + rnorm(p, 0, 0.02)

gen_dat <- function(n, beta) {
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  colnames(X) <- feature_names
  lp <- as.vector(X %*% beta)
  true_time <- 20 * (-log(runif(n)) / exp(lp))^(1/2)
  cens_time <- rexp(n, rate = 1/median(true_time) * 0.3)
  time <- pmin(true_time, cens_time) + 0.001
  status <- as.numeric(true_time <= cens_time)
  return(data.frame(time=time, status=status, as.data.frame(X)))
}

prim_data_raw <- gen_dat(n_prim, beta_prim)
aux_data_raw  <- gen_dat(n_aux, beta_aux)
test_data_raw <- gen_dat(n_test, beta_prim)

X_prim <- as.matrix(prim_data_raw[, feature_names])
X_aux  <- as.matrix(aux_data_raw[, feature_names])
X_test <- as.matrix(test_data_raw[, feature_names])

prim_data <- prim_data_raw; prim_data[, feature_names] <- scale(X_prim)
aux_data <- aux_data_raw;   aux_data[, feature_names] <- scale(X_aux)
test_data <- test_data_raw; test_data[, feature_names] <- scale(X_test)

# --- 4. Helper Functions -----------------------------------------------------

# Helper for Lasso Cox Object
coef_to_coxph <- function(coefs, train_data) {
  active_vars <- names(coefs)[abs(coefs) > 1e-6]
  if(length(active_vars) == 0) {
    fit <- coxph(Surv(time, status) ~ 1, data = train_data, x = TRUE, y = TRUE)
  } else {
    fmla <- as.formula(paste("Surv(time, status) ~", paste(active_vars, collapse = " + ")))
    init_vals <- coefs[active_vars]
    fit <- coxph(fmla, data = train_data, init = init_vals,
                 control = coxph.control(iter.max = 0), x = TRUE, y = TRUE)
  }
  return(fit)
}

eval_performance_extended <- function(coef_est, model_obj, name, test_data) {

  # 1. C-index & AUC
  X_test_mat <- as.matrix(test_data[, feature_names])
  risk_score <- as.vector(X_test_mat %*% coef_est)
  c_index <- survival::concordance(Surv(time, status) ~ risk_score, data=test_data, reverse=TRUE)$concordance

  median_time <- median(test_data$time[test_data$status == 1])
  auc_val <- tryCatch({
    roc_res <- timeROC::timeROC(T = test_data$time, delta = test_data$status,
                                marker = risk_score, cause = 1, weighting = "marginal",
                                times = median_time, iid = FALSE)
    roc_res$AUC[[2]]
  }, error = function(e) NA)

  # 2. IBS via PEC
  pec_res <- tryCatch({
    pec::pec(list("Model" = model_obj), formula = Surv(time, status) ~ 1, data = test_data, verbose = FALSE)
  }, error = function(e) { warning(paste(name, "PEC Error:", e$message)); return(NULL) })
  ibs_val <- if(!is.null(pec_res)) pec::crps(pec_res)[2] else NA

  # 3. Coefficient Metrics (TPR, FPR, MCC)
  threshold <- 1e-6 # Numerical threshold for "Active"

  true_active <- abs(beta_prim) > threshold
  est_active  <- abs(coef_est) > threshold

  tp <- sum(true_active & est_active)
  fp <- sum(est_active & !true_active)
  fn <- sum(true_active & !est_active)
  tn <- sum(!true_active & !est_active)

  tpr <- ifelse((tp + fn) == 0, 0, tp / (tp + fn))
  fpr <- ifelse((fp + tn) == 0, 0, fp / (fp + tn))

  mcc_num <- (as.numeric(tp)*as.numeric(tn)) - (as.numeric(fp)*as.numeric(fn))
  mcc_den <- sqrt(as.numeric(tp+fp)*as.numeric(tp+fn)*as.numeric(tn+fp)*as.numeric(tn+fn))
  mcc <- ifelse(mcc_den==0, 0, mcc_num/mcc_den)

  mse <- mean((coef_est - beta_prim)^2)
  precision <- ifelse((tp + fp) == 0, 0, tp / (tp + fp))
  f1 <- ifelse((precision + tpr) == 0, 0, 2 * precision * tpr / (precision + tpr))

  return(list(Method = name, C_index = c_index, AUC_Median = auc_val, IBS = ibs_val,
              MSE = mse, F1 = f1, MCC = mcc,
              Selected = sum(est_active), TPR = tpr, FPR = fpr))
}

# --- 5. Training Models ------------------------------------------------------

cat("\n>> [Model 1] Lasso (Primary)...\n")
cv_fit_prim <- cv.glmnet(as.matrix(prim_data[, feature_names]), Surv(prim_data$time, prim_data$status), family = "cox", alpha = 1)
coef_lasso_prim <- as.numeric(coef(cv_fit_prim, s = "lambda.min")); names(coef_lasso_prim) <- feature_names
fit_lasso_prim <- coef_to_coxph(coef_lasso_prim, prim_data)
res_lasso_prim <- eval_performance_extended(coef_lasso_prim, fit_lasso_prim, "Lasso (Primary)", test_data)

cat(">> [Model 2] Lasso (Combined)...\n")
comb_data <- rbind(prim_data, aux_data)
cv_fit_comb <- cv.glmnet(as.matrix(comb_data[, feature_names]), Surv(comb_data$time, comb_data$status), family = "cox", alpha = 1)
coef_lasso_comb <- as.numeric(coef(cv_fit_comb, s = "lambda.min")); names(coef_lasso_comb) <- feature_names
fit_lasso_comb <- coef_to_coxph(coef_lasso_comb, comb_data)
res_lasso_comb <- eval_performance_extended(coef_lasso_comb, fit_lasso_comb, "Lasso (Combined)", test_data)

cat(">> [Model 3] TransCox Two-Stage...\n")
final_res <- runTransCox_TwoStage(
  primData = prim_data, auxData = aux_data, cov = feature_names, statusvar = "status",
  p_value_threshold = 0.05, lambda1 = 0.01, lambda2 = 0.01, lambda_beta = 0.05,
  learning_rate = 0.001, nsteps = 5000, auto_tune = FALSE, verbose = FALSE
)
coef_transcox <- final_res$new_beta
fit_transcox <- final_res$fit_obj
res_transcox <- eval_performance_extended(coef_transcox, fit_transcox, "TransCox", test_data)

# --- 6. Export Results -------------------------------------------------------
results_df <- rbind(as.data.frame(res_lasso_prim), as.data.frame(res_lasso_comb), as.data.frame(res_transcox))
# Reorder for clarity
results_df <- results_df[, c("Method", "C_index", "AUC_Median", "IBS", "MSE", "F1", "MCC", "Selected", "TPR", "FPR")]

print(knitr::kable(results_df, digits = 4))
write.csv(results_df, paste0(save_dir, "metrics_summary.csv"), row.names = FALSE)

# Group Summary
group_summary <- data.frame(
  Feature_Group = c("A: Consistent (1-10)", "B: Conflict (11-15)", "C: Source Missing (16-20)", "D: Source Noise (21-25)"),
  Truth_Mean = c(mean(beta_prim[1:10]), mean(beta_prim[11:15]), mean(beta_prim[16:20]), mean(beta_prim[21:25])),
  Lasso_Prim_Mean = c(mean(coef_lasso_prim[1:10]), mean(coef_lasso_prim[11:15]), mean(coef_lasso_prim[16:20]), mean(coef_lasso_prim[21:25])),
  Lasso_Comb_Mean = c(mean(coef_lasso_comb[1:10]), mean(coef_lasso_comb[11:15]), mean(coef_lasso_comb[16:20]), mean(coef_lasso_comb[21:25])),
  TransCox_Mean = c(mean(coef_transcox[1:10]), mean(coef_transcox[11:15]), mean(coef_transcox[16:20]), mean(coef_transcox[21:25]))
)
print(knitr::kable(group_summary, digits = 3))
write.csv(group_summary, file = paste0(save_dir, "coef_group_summary.csv"), row.names = FALSE)


# --- 7. Visualization --------------------------------------------------------

# 7.1 KM Curves
get_km <- function(coefs, title) {
  rs <- as.vector(as.matrix(test_data[,feature_names]) %*% coefs)
  if(var(rs)<1e-9) rs <- rnorm(length(rs))
  grp <- ifelse(rs > median(rs), "High", "Low")

  plot_df <- data.frame(time = test_data$time, status = test_data$status,
                        grp = factor(grp, levels = c("Low", "High")))

  fit <- survfit(Surv(time, status) ~ grp, data = plot_df)

  ggsurvplot(fit, data = plot_df,
             title = title,
             pval = TRUE, conf.int = TRUE,
             palette = c("#2E9FDF", "#E7B800"),
             risk.table = TRUE, risk.table.height = 0.25,
             legend = "top", legend.title = "Risk Group",
             ggtheme = theme_classic())
}

p1 <- get_km(coef_lasso_prim, "Lasso (Primary)")
p2 <- get_km(coef_lasso_comb, "Lasso (Combined)")
p3 <- get_km(coef_transcox, "TransCox")

pdf(paste0(save_dir, "Merged_KM_Curves.pdf"), width = 18, height = 6, onefile = FALSE)
arrange_ggsurvplots(list(p1, p2, p3), ncol = 3, nrow = 1)
dev.off()
cat(">> Saved Merged KM Curves (Whitespace Fixed).\n")

# 7.2 Lollipop Plot (Updated for 5 Noise Vars)
plot_indices <- 1:25 # 20 Active + 5 Noise
plot_data <- data.frame(
  Index = rep(plot_indices, 4),
  Value = c(beta_prim[plot_indices], coef_lasso_prim[plot_indices],
            coef_lasso_comb[plot_indices], coef_transcox[plot_indices]),
  Method = factor(rep(c("Truth", "Lasso (Prim)", "Lasso (Comb)", "TransCox"), each=length(plot_indices)),
                  levels = c("Truth", "Lasso (Prim)", "Lasso (Comb)", "TransCox"))
)

p_coef <- ggplot(plot_data, aes(x = Index, y = Value, color = Method)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray60") +
  geom_point(position = position_dodge(width = 0.7), size = 2) +
  geom_linerange(aes(ymin = 0, ymax = Value), position = position_dodge(width = 0.7)) +

  annotate("rect", xmin=0.5, xmax=10.5, ymin=-Inf, ymax=Inf, alpha=0.1, fill="green") +
  annotate("text", x=5.5, y=1.4, label="A: Consistent", size=3, fontface="bold") +

  annotate("rect", xmin=10.5, xmax=15.5, ymin=-Inf, ymax=Inf, alpha=0.1, fill="red") +
  annotate("text", x=13, y=1.4, label="B: Conflict", size=3, fontface="bold") +

  annotate("rect", xmin=15.5, xmax=20.5, ymin=-Inf, ymax=Inf, alpha=0.1, fill="blue") +
  annotate("text", x=18, y=1.4, label="C: Missing", size=3, fontface="bold") +

  annotate("rect", xmin=20.5, xmax=25.5, ymin=-Inf, ymax=Inf, alpha=0.1, fill="gray") +
  annotate("text", x=23, y=1.4, label="D: Noise", size=3, fontface="bold") +

  scale_color_manual(values = c("black", "gray", "blue", "red")) +
  labs(title = "Coefficient Recovery (Groups A, B, C, D)", y = "Coefficient Value", x = "Variable Index") +
  theme_bw() + theme(legend.position = "bottom")

ggsave(paste0(save_dir, "Coefficient_Lollipop.pdf"), plot = p_coef, width = 16, height = 6)
cat(">> Saved Coefficient Lollipop Plot.\n")

# 7.3 Feature Selection Composition Plot (Strict Template)
cat("\n>> Generating Feature Selection Composition Plot...\n")

# 1. Define true active variable count
n_active_truth <- 20

# 2. Data Preparation
get_counts <- function(coef_vec, method_name) {
  est_act <- abs(coef_vec) > 1e-6
  true_act <- abs(beta_prim) > 1e-6

  tp <- sum(est_act & true_act)
  fp <- sum(est_act & !true_act) # Noise

  return(data.frame(Method = method_name,
                    True_Positives = tp,
                    False_Positives = fp))
}

# Construct plotting data (Method names must match Step 4)
plot_data_comp <- rbind(
  get_counts(coef_lasso_prim, "Lasso (Primary)"),
  get_counts(coef_lasso_comb, "Lasso (Combined)"),
  get_counts(coef_transcox, "TransCox")
)

# Convert to Long Format (requires tidyr)
plot_data_long <- data.frame(
  Method = rep(plot_data_comp$Method, 2),
  Count = c(plot_data_comp$False_Positives, plot_data_comp$True_Positives),
  Variable_Type = rep(c("Noise Variables (FP)", "True Active Variables (TP)"), each=nrow(plot_data_comp))
)

# [CRITICAL] Enforce factor order for plotting stability
target_levels <- c("Lasso (Primary)", "Lasso (Combined)", "TransCox")
plot_data_long$Method <- factor(plot_data_long$Method, levels = target_levels)
plot_data_long$Variable_Type <- factor(plot_data_long$Variable_Type,
                                       levels = c("Noise Variables (FP)", "True Active Variables (TP)"))

# 3. Annotation Data
# Ensure anno_data matches target levels and order
anno_data <- results_df %>%
  filter(Method %in% target_levels) %>%
  select(Method, MCC, TPR, FPR, Selected) %>%
  mutate(Label = sprintf("MCC: %.2f\nTPR: %.2f\nFPR: %.2f", MCC, TPR, FPR))

anno_data$Method <- factor(anno_data$Method, levels = target_levels)

# 4. Plotting (Composition Template)
p_comp <- ggplot(plot_data_long, aes(x = Method, y = Count, fill = Variable_Type)) +
  # Draw bar chart
  geom_col(width = 0.55, alpha = 0.9, color = "black") +

  geom_hline(yintercept = n_active_truth, linetype = "longdash", color = "#333333", size = 0.8) +

  scale_fill_manual(values = c("Noise Variables (FP)" = "#D95F5F",
                               "True Active Variables (TP)" = "#34678C")) +

  scale_y_continuous(
    expand = expansion(mult = c(0, 0.3)), # Reserve space at top
    sec.axis = sec_axis(~ ., breaks = n_active_truth, labels = paste0("True Size(", n_active_truth, ")"))
  ) +

  geom_text(data = anno_data,
            aes(x = Method, y = Selected + 3, label = Label, fill = NULL),
            vjust = 0, size = 3.2, lineheight = 0.9, color = "black") +

  labs(title = "Feature Selection Composition",
       y = "Number of Selected Features",
       x = "",
       fill = "Feature Type") +

  theme_bw() +
  theme(
    legend.position = "top",
    legend.title = element_text(face = "bold", size = 10),
    legend.text = element_text(size = 9),
    legend.background = element_rect(fill = "transparent"),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),

    axis.text.x = element_text(size = 11, color = "black", face = "bold"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.y = element_text(size = 11, face = "bold", margin = margin(r = 10)),

    axis.text.y.right = element_text(color = "#333333", face = "bold.italic", size = 10),
    axis.ticks.y.right = element_line(color = "#333333"),

    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.grid.major.y = element_line(color = "#E5E5E5"),
    panel.grid.minor.y = element_blank(),
    panel.border = element_rect(color = "black", size = 0.8)
  ) +

  geom_text(aes(label = round(Count, 1), color = Variable_Type),
            position = position_stack(vjust = 0.5),
            size = 3.5, fontface = "bold", show.legend = FALSE) +
  scale_color_manual(values = c("Noise Variables (FP)" = "white",
                                "True Active Variables (TP)" = "white"))

ggsave(paste0(save_dir, "Feature_Selection_Composition.pdf"), plot = p_comp, width = 8, height = 6)
cat(">> Saved Feature Selection Composition Plot (Template Corrected).\n")
