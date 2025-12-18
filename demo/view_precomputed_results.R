# Demo: View Pre-computed Simulation Results (Paper Replication)
# Description: Loads and visualizes the results from:
#              1. A representative single run (Quick Start)
#              2. The full Monte Carlo simulation study (100 Runs)
#              No heavy computation is performed in this demo.

library(TransCoxSparse)
library(utils)

# --- Helper Function to Open Files Safely ---
open_file <- function(fpath, desc) {
  if (file.exists(fpath)) {
    cat(sprintf("   [OPEN] %s: %s\n", desc, basename(fpath)))
    try(utils::browseURL(fpath), silent = TRUE)
  } else {
    cat(sprintf("   [MISSING] %s not found.\n", basename(fpath)))
  }
}

# --- Locate the Main Results Directory ---
# Note: "inst/demo_results" moves to "demo_results" at the package root after installation
base_dir <- system.file("demo_results", package = "TransCoxSparse")

if (base_dir == "") {
  stop("Error: 'demo_results' folder not found in the package. Please verify installation.")
}

cat("==============================================================\n")
cat("   TransCox-Sparse: Pre-computed Results Viewer\n")
cat("==============================================================\n")
cat(">> Results Directory located at:", base_dir, "\n")


# ==============================================================================
# PART 1: Quick Start Results (Single Run, ID=45, Seed=168)
# ==============================================================================
cat("\n\n")
cat("--------------------------------------------------------------\n")
cat(" PART 1: Representative Single Run (Quick Start)\n")
cat(" Description: Results from a single split (Seed 168), showing\n")
cat("              KM curves and coefficient estimation details.\n")
cat("--------------------------------------------------------------\n")

quick_start_dir <- file.path(base_dir, "TransCox_Quick_Start_Results")

if (dir.exists(quick_start_dir)) {

  # 1. Display Metrics
  metrics_file <- file.path(quick_start_dir, "metrics_summary.csv")
  if (file.exists(metrics_file)) {
    cat("\n>> Performance Metrics (Single Run):\n")
    print(read.csv(metrics_file))
  }

  # 2. Open Plots
  cat("\n>> Opening Visualizations...\n")
  open_file(file.path(quick_start_dir, "Merged_KM_Curves.pdf"), "KM Curves")
  open_file(file.path(quick_start_dir, "Coefficient_Comparison.pdf"), "Coefficients")

} else {
  cat(">> Warning: Quick Start directory not found.\n")
}


# ==============================================================================
# INTERACTIVE PAUSE
# ==============================================================================
cat("\n")
readline(prompt = ">> Press [Enter] to continue to 100-Run Simulation Results...")


# ==============================================================================
# PART 2: Monte Carlo Simulation (100 Runs)
# ==============================================================================
cat("\n\n")
cat("--------------------------------------------------------------\n")
cat(" PART 2: Monte Carlo Simulation Study (100 Runs)\n")
cat(" Description: Aggregated performance metrics and stability\n")
cat("              analysis across 100 random seeds.\n")
cat("--------------------------------------------------------------\n")

sim_100_dir <- file.path(base_dir, "TransCox_Reproduce_Simulations")

if (dir.exists(sim_100_dir)) {

  # 1. Display Summary Stats
  summary_file <- file.path(sim_100_dir, "Summary_Statistics_Mean_SD.csv")
  if (file.exists(summary_file)) {
    cat("\n>> Pooled Summary Statistics (Mean & SD over 100 runs):\n")
    print(read.csv(summary_file))
  }

  # 2. Open Plots
  cat("\n>> Opening Visualizations...\n")
  open_file(file.path(sim_100_dir, "Boxplot_Cindex.pdf"), "C-index Boxplot")
  open_file(file.path(sim_100_dir, "Selection_Probability.pdf"), "Selection Prob")

} else {
  cat(">> Warning: Simulation directory not found.\n")
}

cat("\n==============================================================\n")
cat("   End of Demonstration\n")
cat("==============================================================\n")
