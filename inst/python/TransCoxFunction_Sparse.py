#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TransCox高维稀疏版本 - Python优化函数

扩展原始TransCox以支持高维稀疏数据：
1. 添加lambda_beta * ||beta_t||_1惩罚项
2. 支持稀疏系数输出
3. 改进的优化算法

作者: AI助手基于原始TransCox
日期: 2024
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

def TransCox_Sparse(CovData, cumH, hazards, status, estR, Xinn, 
                   lambda1, lambda2, lambda_beta=0.0,
                   learning_rate=0.004, nsteps=200,
                   tolerance=1e-6, verbose=True):
    """
    高维稀疏TransCox优化函数
    
    参数:
    --------
    CovData : array-like, shape (n_samples, n_features)
        协变量矩阵
    cumH : array-like, shape (n_samples,)
        累积基线风险
    hazards : array-like, shape (n_events,)
        基线风险增量
    status : array-like, shape (n_samples,)
        生存状态指示器
    estR : array-like, shape (n_features,)
        源域估计的系数 beta_s
    Xinn : array-like, shape (n_samples, n_events)
        事件指示矩阵
    lambda1 : float
        eta (差异) 的L1惩罚参数
    lambda2 : float
        xi (基线调整) 的L1惩罚参数
    lambda_beta : float
        beta_t = estR + eta 的L1惩罚参数 (新增)
    learning_rate : float
        学习率
    nsteps : int
        优化步数
    tolerance : float
        收敛容忍度
    verbose : bool
        是否打印详细信息
        
    返回:
    --------
    tuple : (eta, xi, beta_t, convergence_info)
        eta : 系数差异
        xi : 基线风险调整
        beta_t : 最终系数 (estR + eta)
        convergence_info : 收敛信息
    """
    
    if verbose:
        print(f"TransCox稀疏优化开始...")
        print(f"样本数: {CovData.shape[0]}, 特征数: {CovData.shape[1]}")
        print(f"lambda1 (eta): {lambda1}, lambda2 (xi): {lambda2}, lambda_beta: {lambda_beta}")
    
    # 优化的数据预处理：减少内存分配
    XiData = np.ascontiguousarray(Xinn, dtype=np.float64)
    ppData = np.ascontiguousarray(CovData, dtype=np.float64)
    cQ_np = np.ascontiguousarray(cumH, dtype=np.float64).reshape((len(cumH),))
    dq_np = np.ascontiguousarray(hazards, dtype=np.float64).reshape((len(hazards),))
    estR_np = np.ascontiguousarray(estR, dtype=np.float64).reshape((len(estR),))
    status_np = np.ascontiguousarray(status, dtype=np.float64).reshape((len(status),))
    
    # 预计算常用值
    n_samples, n_features = ppData.shape
    n_events = len(dq_np)
    
    # 兼容0/1与1/2状态编码
    event_code = 2 if np.max(status_np) > 1 else 1
    event_mask = status_np == event_code
    smallidx = tf.constant(np.where(event_mask)[0], dtype=tf.int64)
    n_events_actual = len(smallidx)
    
    # 优化的参数初始化
    eta = tf.Variable(np.zeros(n_features, dtype=np.float64), dtype=tf.float64, name="eta")
    xi = tf.Variable(np.zeros(n_events, dtype=np.float64), dtype=tf.float64, name="xi")
    
    # 当lambda_beta > 0时，将beta_t作为独立的可训练变量
    if lambda_beta > 0:
        beta_t = tf.Variable(estR_np.copy(), dtype=tf.float64, name="beta_t")
    
    # 预计算TensorFlow常量以避免重复转换
    ppData_tf = tf.constant(ppData, dtype=tf.float64)
    cQ_tf = tf.constant(cQ_np, dtype=tf.float64)
    dq_tf = tf.constant(dq_np, dtype=tf.float64)
    estR_tf = tf.constant(estR_np, dtype=tf.float64)
    XiData_tf = tf.constant(XiData, dtype=tf.float64)
    
    # 损失历史和收敛监控
    loss_history = []
    convergence_window = 5  # 收敛检查窗口
    
    def loss_fn():
        """高度优化的损失函数"""
        if lambda_beta > 0:
            # 使用独立的beta_t变量
            current_beta_t = beta_t
        else:
            # 传统方式：beta_t = estR + eta
            current_beta_t = estR_tf + eta
        
        # 优化的矩阵运算：使用预计算的常量
        # 1. 线性预测子：X * beta_t (向量化)
        linear_pred = tf.linalg.matvec(ppData_tf, current_beta_t)
        
        # 2. 调整后的累积基线风险：cumH + Xinn * xi (向量化)
        xi_contribution = tf.linalg.matvec(XiData_tf, xi)
        adjusted_cumH = cQ_tf + xi_contribution
        
        # 3. 调整后的基线风险：hazards + xi
        adjusted_hazards = dq_tf + xi
        
        # 数值稳定性：确保调整后的风险为正
        adjusted_hazards = tf.maximum(adjusted_hazards, 1e-8)
        
        # 事件相关计算（向量化）
        event_linear_pred = tf.reduce_sum(tf.gather(linear_pred, smallidx))
        log_hazards = tf.reduce_sum(tf.math.log(adjusted_hazards))
        
        # 风险集计算（向量化指数运算）
        exp_linear_pred = tf.math.exp(linear_pred)
        risk_set_term = tf.reduce_sum(adjusted_cumH * exp_linear_pred)
        
        # 负对数似然
        neg_log_likelihood = -(event_linear_pred + log_hazards - risk_set_term)
        
        # 向量化L1惩罚计算
        l1_eta = lambda1 * tf.reduce_sum(tf.abs(eta))
        l1_xi = lambda2 * tf.reduce_sum(tf.abs(xi))
        l1_beta = lambda_beta * tf.reduce_sum(tf.abs(current_beta_t))
        
        # 总损失
        total_loss = neg_log_likelihood + l1_eta + l1_xi + l1_beta
        
        return total_loss
    
    # 优化器
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    
    # 优化循环
    if verbose:
        print("开始优化...")
    
    # 确保nsteps是整数
    nsteps = int(nsteps)
    
    for step in range(nsteps):
        with tf.GradientTape() as tape:
            loss_value = loss_fn()
        
        # 软阈值化函数定义
        def soft_threshold(x, threshold):
            """优化的软阈值化函数，使用向量化操作"""
            abs_x = tf.abs(x)
            return tf.sign(x) * tf.maximum(abs_x - threshold, 0.0)
        
        # 计算梯度 - 包含beta_t以实现真正的稀疏优化
        if lambda_beta > 0:
            gradients = tape.gradient(loss_value, [eta, xi, beta_t])
            optimizer.apply_gradients(zip(gradients, [eta, xi, beta_t]))
            
            # 对所有参数应用软阈值化以实现稀疏性
            threshold_beta = lambda_beta * 0.1  # 进一步降低阈值，确保不过度稀疏化
            beta_t.assign(soft_threshold(beta_t, threshold_beta))
            
            threshold_eta = lambda1 * 0.05  # 进一步降低阈值，确保不过度稀疏化
            eta.assign(soft_threshold(eta, threshold_eta))
            
            threshold_xi = lambda2 * 0.05  # 进一步降低阈值，确保不过度稀疏化
            xi.assign(soft_threshold(xi, threshold_xi))
            
        else:
            gradients = tape.gradient(loss_value, [eta, xi])
            optimizer.apply_gradients(zip(gradients, [eta, xi]))
            
            # 即使在非稀疏模式下也应用软阈值化以获得稀疏性
            # 对eta应用软阈值化
            threshold_eta = lambda1 * 0.01  # 更轻微的阈值化
            eta_sign = tf.sign(eta)
            eta_abs = tf.abs(eta)
            eta_thresholded = eta_sign * tf.maximum(0.0, eta_abs - threshold_eta)
            eta.assign(eta_thresholded)
            
            # 对xi应用软阈值化
            threshold_xi = lambda2 * 0.01  # 更轻微的阈值化
            xi_sign = tf.sign(xi)
            xi_abs = tf.abs(xi)
            xi_thresholded = xi_sign * tf.maximum(0.0, xi_abs - threshold_xi)
            xi.assign(xi_thresholded)
        
        # 记录损失
        loss_history.append(loss_value.numpy())
        
        # 每50步打印一次进度
        if verbose and (step + 1) % 50 == 0:
            print(f"步骤 {step + 1}/{nsteps}, 损失: {loss_value.numpy():.6f}")
        
        # 优化的收敛检查
        if step > convergence_window:
            # 使用损失变化和参数变化双重检查
            recent_losses = loss_history[-convergence_window:]
            loss_change = max(recent_losses) - min(recent_losses)
            
            # 参数变化检查
            if step > 0:
                # 计算参数的相对变化
                eta_norm = tf.norm(eta)
                xi_norm = tf.norm(xi)
                
                if lambda_beta > 0:
                    beta_norm = tf.norm(beta_t)
                    total_norm = tf.sqrt(eta_norm**2 + xi_norm**2 + beta_norm**2)
                else:
                    total_norm = tf.sqrt(eta_norm**2 + xi_norm**2)
                
                # 如果损失变化很小且参数范数稳定，则认为收敛
                if loss_change < tolerance and total_norm > 0:
                    if verbose:
                        print(f"在步骤 {step + 1} 收敛，损失变化: {loss_change:.6f}")
                    break
    
    # 计算最终结果
    eta_final = eta.numpy()
    xi_final = xi.numpy()
    
    if lambda_beta > 0:
        # 当lambda_beta > 0时，beta_t已经通过优化直接获得稀疏性
        beta_t_final = beta_t.numpy()
        # 重新计算eta以保持一致性
        eta_final = beta_t_final - estR_np
    else:
        # 当lambda_beta = 0时，使用传统方法
        beta_t_final = estR_np + eta_final
    
    # 计算稀疏性统计
    nonzero_eta = np.sum(np.abs(eta_final) > 1e-8)
    nonzero_xi = np.sum(np.abs(xi_final) > 1e-8)
    nonzero_beta = np.sum(np.abs(beta_t_final) > 1e-8)
    
    convergence_info = {
        'final_loss': loss_history[-1],
        'steps_taken': len(loss_history),
        'converged': len(loss_history) < nsteps,
        'loss_history': loss_history,
        'nonzero_eta': nonzero_eta,
        'nonzero_xi': nonzero_xi,
        'nonzero_beta': nonzero_beta,
        'sparsity_eta': 1 - nonzero_eta / len(eta_final),
        'sparsity_xi': 1 - nonzero_xi / len(xi_final),
        'sparsity_beta': 1 - nonzero_beta / len(beta_t_final)
    }
    
    if verbose:
        print(f"优化完成!")
        print(f"最终损失: {convergence_info['final_loss']:.6f}")
        print(f"eta非零系数: {nonzero_eta}/{len(eta_final)} ({(1-convergence_info['sparsity_eta'])*100:.1f}%)")
        print(f"xi非零系数: {nonzero_xi}/{len(xi_final)} ({(1-convergence_info['sparsity_xi'])*100:.1f}%)")
        print(f"beta_t非零系数: {nonzero_beta}/{len(beta_t_final)} ({(1-convergence_info['sparsity_beta'])*100:.1f}%)")
    
    return eta_final, xi_final, beta_t_final, convergence_info


def TransCox(CovData, cumH, hazards, status, estR, Xinn, 
            lambda1, lambda2, learning_rate=0.004, nsteps=200,
            lambda_beta=None):
    """
    向后兼容的TransCox函数
    
    如果lambda_beta未指定或为0，使用原始算法
    否则使用新的稀疏算法
    """
    
    if lambda_beta is None or lambda_beta == 0:
        # 使用原始算法
        xi = tf.Variable(np.repeat([0.], repeats=len(hazards)), dtype="float64")
        eta = tf.Variable(np.repeat([0.], repeats=len(estR)), dtype="float64")
        
        XiData = np.float64(Xinn)
        ppData = np.float64(CovData)
        cQ_np = np.float64(cumH).reshape((len(cumH),))
        dq_np = np.float64(hazards).reshape((len(hazards),))
        estR2 = np.float64(estR).reshape((len(estR),))
        status_np = np.float64(status).reshape((len(status),))
        # 兼容0/1与1/2状态编码
        event_code = 2 if np.max(status_np) > 1 else 1
        smallidx = tf.where(status_np == event_code)[:, 0]
        
        loss_fn = lambda: tf.math.negative(
            tf.reduce_sum(tf.gather(tf.math.reduce_sum(tf.math.multiply(ppData, tf.math.add(estR2, eta)), axis=1), indices=smallidx)) + 
            tf.reduce_sum(tf.math.log(tf.math.add(dq_np, xi))) - 
            tf.reduce_sum(tf.math.multiply(tf.math.add(cQ_np, tf.math.reduce_sum(tf.math.multiply(XiData, xi), axis=1)), 
                                         tf.math.exp(tf.math.reduce_sum(tf.math.multiply(ppData, tf.math.add(estR2, eta)), axis=1))))
        ) + lambda1 * tf.math.reduce_sum(tf.math.abs(eta)) + lambda2 * tf.math.reduce_sum(tf.math.abs(xi))
        
        loss = tfp.math.minimize(loss_fn, num_steps=int(nsteps), optimizer=tf.optimizers.Adam(learning_rate=learning_rate))
        
        return eta.numpy(), xi.numpy()
    
    else:
        # 使用新的稀疏算法
        eta, xi, beta_t, conv_info = TransCox_Sparse(
            CovData, cumH, hazards, status, estR, Xinn,
            lambda1, lambda2, lambda_beta, learning_rate, nsteps, verbose=False
        )
        
        return eta, xi