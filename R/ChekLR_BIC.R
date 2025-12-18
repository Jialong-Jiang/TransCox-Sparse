#' Learning Rate Check and Selection based on BIC
#' 
#' @description 
#' Selects the optimal learning rate using the BIC criterion, while keeping other parameters fixed.
#' 
#' @param primData Target domain data.
#' @param auxData Source domain data.
#' @param cov Vector of covariate names.
#' @param statusvar Name of the status variable.
#' @param lr_vec Vector of candidate learning rate values.
#' @param l1 lambda1 parameter (fixed).
#' @param l2 lambda2 parameter (fixed).
#' @param nsteps Number of optimization steps.
#' 
#' @return A list containing the optimal learning rate and BIC values.
#' @export
CheckLR_BIC <- function(primData, auxData, cov = c("X1", "X2"),
                        statusvar = "status",
                        lr_vec = seq(0.001, 0.004, by = 0.0005),
                        l1 = 0.1, l2 = 0.1,
                        nsteps = 200) {

    TransCox <- NULL
    # Load Python function if not already loaded
    if (!exists("TransCox")) {
        py_file <- system.file("python", "TransCoxFunction.py", package = "TransCox")
        if (py_file == "") py_file <- file.path(getwd(), "inst", "python", "TransCoxFunction.py")
        if (file.exists(py_file)) {
            reticulate::source_python(py_file)
        }
    }

    Cout <- GetAuxSurv(auxData, cov = cov)
    Pout <- GetPrimaryParam(primData, q = Cout$q, estR = Cout$estR)

    CovData = Pout$primData[, cov]
    status = Pout$primData[, statusvar]
    cumH = Pout$primData$fullCumQ
    hazards = Pout$dQ$dQ

    BICvec <- rep(NA, length(lr_vec))
    for(i in 1:length(lr_vec)) {
        test <- TransCox(CovData = as.matrix(CovData),
                         cumH = cumH,
                         hazards = hazards,
                         status = status,
                         estR = Pout$estR,
                         Xinn = Pout$Xinn,
                         lambda1 = l1, lambda2 = l2,
                         learning_rate = lr_vec[i],
                         nsteps = nsteps)
        names(test) <- c("eta", "xi")
        newBeta = Pout$estR + test$eta
        newHaz = Pout$dQ$dQ + test$xi

        BICvalue <- GetBIC(status = status,
                           CovData = CovData,
                           hazards = hazards,
                           newBeta = newBeta,
                           newHaz = newHaz,
                           eta = test$eta,
                           xi = test$xi,
                           cutoff = 1e-5)
        BICvec[i] <- BICvalue
    }
    names(BICvec) = lr_vec

    idx0 <- which.min(BICvec)

    b_lr <- lr_vec[idx0]

    return(list(best_lr = b_lr,
                BICvec = BICvec))
}
