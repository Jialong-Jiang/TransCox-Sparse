#' Run Bootstrap TransCox Analysis
#' 
#' @description 
#' Evaluates the stability and confidence intervals of the TransCox model using the Bootstrap method.
#' 
#' @param aData Source domain data.
#' @param pData Target domain data.
#' @param Tres Time points.
#' @param learning_rate Learning rate.
#' @param nsteps Number of optimization steps.
#' @param BestLam1 Optimal lambda1 parameter.
#' @param BestLam2 Optimal lambda2 parameter.
#' @param nbootstrap Number of Bootstrap iterations.
#' @param trueBeta True parameter values (for evaluation).
#' @param Hrange Time range for cumulative hazard function evaluation.
#' @param cov Vector of covariate names.
#' 
#' @return A list containing Bootstrap results.
#' @export
runBtsp_transCox <- function(aData,
                             pData,
                             Tres,
                             learning_rate = 0.002,
                             nsteps = 200,
                             BestLam1 = 0.1,
                             BestLam2 = 0.1,
                             nbootstrap = 100,
                             trueBeta = c(-0.5, 0.5),
                             Hrange = seq(0.15, 2, length.out = 50),
                             cov = c("X1", "X2")) {

    rec_beta = rec_twop_cumH = rec_twop_surv <- matrix(NA, nbootstrap, 2)
    rec_cumH <- matrix(NA, nbootstrap, 50)
    rec_survTrans = matrix(NA, nbootstrap, 50)

    pb = txtProgressBar(min = 0, max = nbootstrap, style = 2, initial = 0)
    for(i in 1:nbootstrap) {
        setTxtProgressBar(pb, i)

        aData_btsp <- aData[sample(1:nrow(aData), nrow(aData), replace = TRUE), ]
        pData_btsp <- pData[sample(1:nrow(pData), nrow(pData), replace = TRUE), ]
        Cout_btsp <- GetAuxSurv(aData_btsp, cov = cov)
        Pout_btsp <- GetPrimaryParam(pData_btsp, q = Cout_btsp$q, estR = Cout_btsp$estR)
        Tres_btsp <- runTransCox_one(Pout_btsp,
                                     l1 = BestLam1,
                                     l2 = BestLam2,
                                     learning_rate = learning_rate,
                                     nsteps = nsteps,
                                     cov = cov)
        Htime <- Tres_btsp$time
        DupN <- sum(duplicated(Htime))
        myHvec <- data.frame(hazard = dQtocumQ(Tres_btsp$new_IntH), time = Htime)
        ttt1 <- stats::approx(x = myHvec$time, y = myHvec$hazard, xout = Hrange)$y

        rec_beta[i,] <- Tres_btsp$new_beta
        rec_cumH[i,] <- ttt1
    }
    close(pb)
    return(list(rec_beta = rec_beta,
                beta_SE_btsp = apply(rec_beta, 2, stats::sd),
                rec_cumH = rec_cumH,
                cumH_SE_btsp = apply(rec_cumH, 2, stats::sd)))
}
