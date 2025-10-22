#' 基于BIC的学习率检查和选择
#' 
#' @description 
#' 使用BIC准则选择最优学习率，固定其他参数
#' 
#' @param primData 目标域数据
#' @param auxData 源域数据
#' @param cov 协变量名称向量
#' @param statusvar 状态变量名称
#' @param lr_vec 学习率候选值向量
#' @param l1 lambda1参数（固定）
#' @param l2 lambda2参数（固定）
#' @param nsteps 优化步数
#' 
#' @return 包含最优学习率和BIC值的列表
#' @export
CheckLR_BIC <- function(primData, auxData, cov = c("X1", "X2"),
                        statusvar = "status",
                        lr_vec = seq(0.001, 0.004, by = 0.0005),
                        l1 = 0.1, l2 = 0.1,
                        nsteps = 200) {
    # require("reticulate")
    # source_python('./python/TransCoxFunction.py')

    TransCox <- NULL
    .onLoad <- function(libname, pkgname) {
        reticulate::py_run_file(system.file("python", "TransCoxFunction.py", package = "TransCox"))
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
