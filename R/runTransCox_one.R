#' 运行单次TransCox模型训练
#' 
#' @description 
#' 使用Python后端运行单次TransCox模型训练，估计eta和xi参数
#' 
#' @param Pout 包含目标域数据和源域参数的预处理结果
#' @param l1 eta的L1惩罚参数
#' @param l2 xi的L1惩罚参数  
#' @param learning_rate 学习率
#' @param nsteps 优化步数
#' @param cov 协变量名称向量
#' 
#' @return 包含eta和xi的列表
#' @export
runTransCox_one <- function(Pout, l1 = 1, l2 = 1, learning_rate = 0.004, nsteps = 200,
                            cov = c('X1', 'X2')){
    # require("reticulate")
    # source_python('/Users/zli16/Dropbox/TransCox/TransCox_package/TransCox/python/TransCoxFunction.py')

    TransCox <- NULL
    .onLoad <- function(libname, pkgname) {
        tf <<- reticulate::import("tensorflow", delay_load = TRUE)
        tfp <<- reticulate::import("tensorflow_probability", delay_load = TRUE)
        np <<- reticulate::import("numpy", delay_load = TRUE)
        if (!exists("TransCox")) {
    reticulate::source_python(system.file("python", "TransCoxFunction.py", package = "TransCox"))
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
