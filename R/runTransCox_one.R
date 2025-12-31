#' Run Single TransCox Model Training
#'
#' @description
#' Runs a single iteration of TransCox model training using the Python backend to estimate eta and xi parameters.
#'
#' @param Pout Preprocessed result containing target domain data and source domain parameters.
#' @param l1 L1 penalty parameter for eta.
#' @param l2 L1 penalty parameter for xi.
#' @param learning_rate Learning rate.
#' @param nsteps Number of optimization steps.
#' @param cov Vector of covariate names.
#'
#' @return A list containing eta and xi.
#' @export
runTransCox_one <- function(Pout, l1 = 1, l2 = 1, learning_rate = 0.004, nsteps = 200,
                            cov = c('X1', 'X2')){

    # Load Python function if not already loaded
    if (!exists("TransCox")) {
         py_file <- system.file("python", "TransCoxFunction.py", package = "TransCox")
         if (py_file == "") py_file <- file.path(getwd(), "inst", "python", "TransCoxFunction.py")
         if (file.exists(py_file)) {
             reticulate::source_python(py_file)
         }
    }

    CovData = Pout$primData[, cov]
    status = Pout$primData[, "status"]
    cumH = Pout$primData$fullCumQ
    hazards = Pout$dQ$dQ

    test <- TransCox(CovData = as.matrix(CovData),
                     cumH = cumH,
                     hazards = hazards,
                     status = status,
                     estR = Pout$estR,
                     Xinn = Pout$Xinn,
                     lambda1 = l1, lambda2 = l2,
                     learning_rate = learning_rate,
                     nsteps = nsteps)
    names(test) <- c("eta", "xi")

    return(list(eta = test$eta,
                xi = test$xi,
                new_beta = Pout$estR + test$eta,
                new_IntH = Pout$dQ$dQ + test$xi,
                time = Pout$primData[status == 2, "time"]))
}
