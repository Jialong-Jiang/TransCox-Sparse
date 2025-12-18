#' Parameter Selection Based on BIC
#'
#' @description
#' Use the Bayesian Information Criterion (BIC) to select the optimal regularization parameters
#' (lambda1, lambda2, and optionally lambda_beta) for the TransCox model.
#' This function serves as a wrapper that automatically selects between sparse and dense implementations.
#'
#' @param primData Target-domain data frame.
#' @param auxData Source-domain data frame.
#' @param cov Vector of covariate names. Default is c("X1", "X2").
#' @param statusvar Name of the status variable. Default is "status".
#' @param lambda1_vec Vector of candidate values for lambda1 (L1 penalty for eta).
#' @param lambda2_vec Vector of candidate values for lambda2 (L1 penalty for xi).
#' @param lambda_beta_vec Vector of candidate values for lambda_beta (L1 penalty for beta).
#'   Only used if \code{use_sparse = TRUE}.
#' @param learning_rate Learning rate for the optimization algorithm. Default is 0.004.
#' @param nsteps Number of optimization steps. Default is 100.
#' @param use_sparse Logical. Whether to use the sparse implementation.
#'   If NULL, it is automatically detected based on data dimensions.
#' @param ... Additional arguments passed to internal functions.
#'
#' @return A list containing the optimal parameters and the BIC matrix.
#'
#' @importFrom utils txtProgressBar setTxtProgressBar
#' @importFrom reticulate source_python
#' @export
SelParam_By_BIC <- function(primData, auxData, cov = c("X1", "X2"),
                            statusvar = "status",
                            lambda1_vec = c(0.1, 0.5, seq(1, 10, by = 0.5)),
                            lambda2_vec = c(0.1, 0.5, seq(1, 10, by = 0.5)),
                            lambda_beta_vec = c(0, 0.001, 0.01, 0.1),
                            learning_rate = 0.004,
                            nsteps = 100,
                            use_sparse = NULL,
                            ...) {

  # 1. Automatic detection of sparse mode
  if (is.null(use_sparse)) {
    n_samples <- nrow(primData)
    n_features <- length(cov)
    # Heuristic: Use sparse if p > 20
    use_sparse <- (n_features > 20)
  }

  # 2. If sparse, delegate to the sparse implementation
  if (use_sparse) {
    # Check if the sparse function is available (it should be in the same package)
    if (exists("SelParam_By_BIC_Sparse")) {
      return(SelParam_By_BIC_Sparse(
        primData = primData,
        auxData = auxData,
        cov = cov,
        statusvar = statusvar,
        lambda1_vec = lambda1_vec,
        lambda2_vec = lambda2_vec,
        lambda_beta_vec = lambda_beta_vec,
        learning_rate = learning_rate,
        nsteps = nsteps,
        use_sparse = TRUE,
        ...
      ))
    } else {
      warning("SelParam_By_BIC_Sparse not found, falling back to dense implementation.")
    }
  }

  # 3. Dense Implementation (Legacy Logic)

  # Load Python function if needed
  if (!exists("TransCox")) {
    py_file <- system.file("python", "TransCoxFunction.py", package = "TransCoxSparse")
    # Fallback for dev mode
    if (py_file == "") py_file <- file.path(getwd(), "inst", "python", "TransCoxFunction.py")

    if (file.exists(py_file)) {
      reticulate::source_python(py_file)
    } else {
      stop("TransCoxFunction.py not found.")
    }
  }

  # Get Parameters
  Cout <- GetAuxSurv(auxData, cov = cov)
  Pout <- GetPrimaryParam(primData, q = Cout$q, estR = Cout$estR)

  CovData <- Pout$primData[, cov]
  status <- Pout$primData[, statusvar]
  cumH <- Pout$primData$fullCumQ
  hazards <- Pout$dQ$dQ

  # Initialize Grid Search
  BICmat <- matrix(NA, length(lambda1_vec), length(lambda2_vec))
  pb <- utils::txtProgressBar(min = 0, max = (length(lambda1_vec)) * length(lambda2_vec), style = 3, initial = 0)
  counter <- 0

  # Grid Search Loop
  for(i in 1:length(lambda1_vec)) {
    for(j in 1:length(lambda2_vec)) {

      counter <- counter + 1
      utils::setTxtProgressBar(pb, counter)

      lambda1 <- lambda1_vec[i]
      lambda2 <- lambda2_vec[j]

      # Run Python TransCox (Dense)
      test <- TransCox(
        CovData = as.matrix(CovData),
        cumH = cumH,
        hazards = hazards,
        status = status,
        estR = Pout$estR,
        Xinn = Pout$Xinn,
        lambda1 = lambda1,
        lambda2 = lambda2,
        learning_rate = learning_rate,
        nsteps = nsteps
      )

      # Python returns list, map to names
      # Assuming test[[1]] is eta, test[[2]] is xi based on python definition
      eta <- test[[1]]
      xi <- test[[2]]

      newBeta <- Pout$estR + eta
      newHaz <- Pout$dQ$dQ + xi

      # Calculate BIC
      BICvalue <- GetBIC(
        status = status,
        CovData = CovData,
        hazards = hazards,
        newBeta = newBeta,
        newHaz = newHaz,
        eta = eta,
        xi = xi,
        cutoff = 1e-5,
        lambda1 = lambda1,
        lambda2 = lambda2,
        lambda_beta = NULL
      )

      BICmat[i,j] <- BICvalue
    }
  }
  close(pb)

  # Format Output
  rownames(BICmat) <- lambda1_vec
  colnames(BICmat) <- lambda2_vec

  idx0 <- which(BICmat == min(BICmat, na.rm = TRUE), arr.ind = TRUE)
  # Handle case with multiple minima
  if(nrow(idx0) > 1) idx0 <- idx0[1, , drop=FALSE]

  b_lambda1 <- lambda1_vec[idx0[1]]
  b_lambda2 <- lambda2_vec[idx0[2]]

  return(list(
    best_la1 = b_lambda1,
    best_la2 = b_lambda2,
    BICmat = BICmat
  ))
}
