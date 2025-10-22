#' 生存分析评估指标
#' 
#' @description 
#' 实现生存分析常用的评估指标，包括C-index、AUC、Brier Score等
#' 

#' Calculate C-index (Concordance Index)
#'
#' @description
#' Computes the concordance index (C-index) for survival models, which measures
#' the proportion of concordant pairs among all comparable pairs of observations.
#'
#' @param predicted_risk Numeric vector of predicted risk scores. Higher values
#'   indicate higher risk of event occurrence.
#' @param time Numeric vector of survival times.
#' @param status Numeric vector of event indicators (1 = event occurred, 0 = censored).
#' 
#' @return Numeric value representing the C-index (range: 0 to 1, where 0.5 indicates
#'   random prediction and 1.0 indicates perfect prediction).
#' 
#' @examples
#' \dontrun{
#' # Example with simulated data
#' time <- c(1, 2, 3, 4, 5)
#' status <- c(1, 1, 0, 1, 0)
#' risk <- c(0.8, 0.6, 0.4, 0.7, 0.3)
#' cindex <- calculate_cindex(risk, time, status)
#' }
#' 
#' @export
calculate_cindex <- function(predicted_risk, time, status) {
    
    if (length(predicted_risk) != length(time) || length(time) != length(status)) {
        stop("输入向量长度不一致")
    }
    
    # 确保 'survival' 包已经安装
    if (!requireNamespace("survival", quietly = TRUE)) {
        stop("请先安装 'survival' 包 (install.packages('survival'))")
    }
    
    # 创建Surv对象
    y <- survival::Surv(time, status)
    
    # 使用 survival::concordance 计算 C-index
    # 这里的公式 y ~ predicted_risk 假设 "predicted_risk" 越高，风险越大
    out <- survival::concordance(y ~ predicted_risk)
    
    # 提取 C-index 值
    return(as.numeric(out$concordance))
}

#' Calculate Time-Dependent AUC
#' 
#' @description
#' Computes the time-dependent Area Under the Curve (AUC) for survival models
#' at a specific time point using the timeROC package.
#' 
#' @param predicted_risk Numeric vector of predicted risk scores. Higher values
#'   indicate higher risk of event occurrence.
#' @param time Numeric vector of survival times.
#' @param status Numeric vector of event indicators (1 = event occurred, 0 = censored).
#' @param time_point Numeric. The specific time point at which to evaluate the AUC.
#' 
#' @return Numeric value representing the time-dependent AUC at the specified time point.
#' 
#' @examples
#' \dontrun{
#' # Example with simulated data
#' time <- c(1, 2, 3, 4, 5)
#' status <- c(1, 1, 0, 1, 0)
#' risk <- c(0.8, 0.6, 0.4, 0.7, 0.3)
#' auc <- calculate_time_dependent_auc(risk, time, status, time_point = 3)
#' }
#' 
#' @export
calculate_time_dependent_auc <- function(predicted_risk, time, status, time_point) {
    
    if (length(predicted_risk) != length(time) || length(time) != length(status)) {
        stop("输入向量长度不一致")
    }
    if (!requireNamespace("timeROC", quietly = TRUE)) {
        stop("请先安装 'timeROC' 包 (install.packages('timeROC'))")
    }
    
    # 使用 timeROC::timeROC 计算
    roc_obj <- timeROC::timeROC(
        T = time,
        delta = status,
        marker = predicted_risk,   # 风险分数越大风险越高
        cause = 1,
        weighting = "marginal",  # 默认的 "marginal" 权重
        times = time_point,
        ROC = FALSE              # 我们只需要AUC，不需要完整的ROC曲线
    )
    
    # timeROC 可能会返回多个时间点（如果time_point不在观测时间内）
    # 我们选取最接近我们请求的 time_point 的那个AUC值
    idx <- which.min(abs(roc_obj$times - time_point))
    
    return(as.numeric(roc_obj$AUC[idx]))
}

#' 计算Brier Score
#' 
#' @param predicted_survival 在 time_point 预测的生存概率 (一个向量)
#' @param time 生存时间
#' @param status 事件指示器
#' @param time_point 评估的时间点
#' @return Brier Score值
#' @export
calculate_brier_score <- function(predicted_survival, time, status, time_point) {
    
    if (length(predicted_survival) != length(time) || length(time) != length(status)) {
        stop("输入向量长度不一致")
    }
    if (!requireNamespace("ipred", quietly = TRUE)) {
        stop("请先安装 'ipred' 包 (install.packages('ipred'))")
    }
    if (!requireNamespace("survival", quietly = TRUE)) {
        stop("请先安装 'survival' 包")
    }
    
    # 确保 time_point 是单一值
    if (length(time_point) > 1) {
        warning("time_point 有多个值，只使用第一个值")
        time_point <- time_point[1]
    }

    # 创建Surv对象
    y <- survival::Surv(time, status)
    
    # ipred::sbrier 的 'pred' 参数需要一个矩阵
    # 矩阵的行对应y中的样本，列对应btime中的时间点
    # 因为我们只评估一个时间点，所以创建一个单列矩阵
    pred_matrix <- matrix(predicted_survival, ncol = 1)
    
    # 使用 ipred::sbrier 计算
    # btime 参数指定了要评估的时间点
    bs <- ipred::sbrier(obj = y, pred = pred_matrix, btime = time_point)
    
    # sbrier 返回的是一个向量，对应 btime 中的每个时间点
    # 因为我们只输入了一个时间点，所以结果就是 bs[1]
    return(as.numeric(bs[1]))
}

#' 计算参数估计的准确性指标
#' 
#' @param estimated_beta 估计的回归系数
#' @param true_beta 真实的回归系数
#' @return 包含多个准确性指标的列表
#' @export
calculate_parameter_accuracy <- function(estimated_beta, true_beta) {
    
    if (length(estimated_beta) != length(true_beta)) {
        stop("估计系数和真实系数长度不一致")
    }
    
    # 均方误差 (MSE)
    mse <- mean((estimated_beta - true_beta)^2)
    
    # 均方根误差 (RMSE)
    rmse <- sqrt(mse)
    
    # 平均绝对误差 (MAE)
    mae <- mean(abs(estimated_beta - true_beta))
    
    # 相关系数
    correlation <- cor(estimated_beta, true_beta, use = "complete.obs")
    
    # 稀疏性相关指标
    true_active <- which(true_beta != 0)
    estimated_active <- which(abs(estimated_beta) > 1e-6)  # 考虑数值精度
    
    # 真正例 (True Positives)
    tp <- length(intersect(true_active, estimated_active))
    
    # 假正例 (False Positives)
    fp <- length(setdiff(estimated_active, true_active))
    
    # 假负例 (False Negatives)
    fn <- length(setdiff(true_active, estimated_active))
    
    # 真负例 (True Negatives)
    tn <- length(true_beta) - tp - fp - fn
    
    # 精确率 (Precision)
    precision <- if (tp + fp > 0) tp / (tp + fp) else 0
    
    # 召回率 (Recall/Sensitivity)
    recall <- if (tp + fn > 0) tp / (tp + fn) else 0
    
    # F1分数
    f1_score <- if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0
    
    # 特异性 (Specificity)
    specificity <- if (tn + fp > 0) tn / (tn + fp) else 0
    
    return(list(
        mse = mse,
        rmse = rmse,
        mae = mae,
        correlation = correlation,
        precision = precision,
        recall = recall,
        f1_score = f1_score,
        specificity = specificity,
        true_positives = tp,
        false_positives = fp,
        false_negatives = fn,
        true_negatives = tn,
        n_estimated_active = length(estimated_active),
        n_true_active = length(true_active)
    ))
}

#' 综合评估模型性能
#' 
#' @param predicted_risk 预测的风险分数
#' @param estimated_beta 估计的回归系数
#' @param test_data 测试数据
#' @param true_beta 真实的回归系数
#' @param time_points 评估的时间点向量
#' @return 包含所有评估指标的列表
#' @export
comprehensive_evaluation <- function(predicted_risk, estimated_beta, test_data, true_beta, time_points = NULL) {
    
    time <- test_data$time
    status <- test_data$status
    
    # 计算C-index
    if (length(unique(predicted_risk)) == 1) {
        warning("所有预测风险值相同，C-index可能不准确")
        return(0.5)
    }
    
    if (sum(status) == 0) {
        warning("没有观察到事件，无法计算C-index")
        return(NA)
    }
    cindex <- calculate_cindex(predicted_risk, time, status)
    
    # 参数准确性
    param_accuracy <- NULL
    if (!is.null(estimated_beta) && !is.null(true_beta)) {
        # 确保系数长度一致
        if (length(estimated_beta) == length(true_beta)) {
            param_accuracy <- calculate_parameter_accuracy(estimated_beta, true_beta)
        } else {
            # 如果长度不一致，只计算基本统计
            param_accuracy <- list(
                estimated_length = length(estimated_beta),
                true_length = length(true_beta),
                estimated_nonzero = sum(abs(estimated_beta) > 1e-6),
                true_nonzero = sum(abs(true_beta) > 1e-6),
                note = "系数长度不匹配，无法计算详细准确性指标"
            )
        }
    }
    
    # 时间依赖的评估指标
    time_dependent_metrics <- NULL
    if (!is.null(time_points)) {
        time_dependent_metrics <- list()
        for (t in time_points) {
            if (t < max(time[status == 1])) {  # 确保时间点有意义
                auc_t <- calculate_time_dependent_auc(predicted_risk, time, status, t)
                time_dependent_metrics[[paste0("AUC_", t)]] <- auc_t
            }
        }
    }
    
    # 预测风险的分布统计
    risk_stats <- list(
        mean_risk = mean(predicted_risk),
        sd_risk = sd(predicted_risk),
        min_risk = min(predicted_risk),
        max_risk = max(predicted_risk),
        range_risk = max(predicted_risk) - min(predicted_risk)
    )
    
    # 创建摘要
    summary_stats <- list(cindex = round(cindex, 4))
    
    if (!is.null(param_accuracy) && !is.null(param_accuracy$correlation)) {
        summary_stats$correlation <- round(param_accuracy$correlation, 4)
        summary_stats$f1_score <- round(param_accuracy$f1_score, 4)
        summary_stats$precision <- round(param_accuracy$precision, 4)
        summary_stats$recall <- round(param_accuracy$recall, 4)
    } else {
        summary_stats$note <- "系数长度不匹配，无法计算参数准确性指标"
    }
    
    return(list(
        cindex = cindex,
        parameter_accuracy = param_accuracy,
        time_dependent_metrics = time_dependent_metrics,
        risk_statistics = risk_stats,
        summary = summary_stats
    ))
}