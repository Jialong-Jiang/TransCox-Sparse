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
    
    # 初始化变量
    xi = tf.Variable(np.repeat([0.], repeats=len(hazards)), dtype="float64", name="xi")
    eta = tf.Variable(np.repeat([0.], repeats=len(estR)), dtype="float64", name="eta")
    
    # 当lambda_beta > 0时，将beta_t作为独立的可训练变量
    if lambda_beta > 0:
        beta_t = tf.Variable(estR.copy(), dtype="float64", name="beta_t")
    
    # 转换数据类型
    XiData = np.float64(Xinn)
    ppData = np.float64(CovData)
    cQ_np = np.float64(cumH).reshape((len(cumH),))
    dq_np = np.float64(hazards).reshape((len(hazards),))
    estR_np = np.float64(estR).reshape((len(estR),))
    status_np = np.float64(status).reshape((len(status),))
    
    # 找到事件发生的样本索引
    smallidx = tf.where(status_np == 2)[:, 0]  # 提取第一列索引
    
    # 存储损失历史用于收敛检查
    loss_history = []
    
    def loss_fn():
        """
        扩展的损失函数，包含beta_t的L1惩罚
        
        原始损失: -log_likelihood + lambda1*||eta||_1 + lambda2*||xi||_1
        新损失: -log_likelihood + lambda1*||eta||_1 + lambda2*||xi||_1 + lambda_beta*||beta_t||_1
        
        其中 beta_t = estR + eta
        """
        
        # 计算当前的beta_t
        if lambda_beta > 0:
            # 使用独立的beta_t变量
            current_beta_t = beta_t
        else:
            # 使用传统方法计算beta_t，确保类型一致
            current_beta_t = tf.math.add(tf.constant(estR_np, dtype="float64"), eta)
        
        # 计算线性预测子
        linear_pred = tf.math.reduce_sum(tf.math.multiply(tf.constant(ppData, dtype="float64"), current_beta_t), axis=1)
        
        # 计算调整后的基线风险，确保数值稳定性
        adjusted_hazards = tf.math.add(tf.constant(dq_np, dtype="float64"), xi)
        adjusted_hazards = tf.maximum(adjusted_hazards, 1e-10)  # 确保为正数
        
        # 计算调整后的累积风险
        adjusted_cumH = tf.math.add(tf.constant(cQ_np, dtype="float64"), 
                                   tf.math.reduce_sum(tf.math.multiply(tf.constant(XiData, dtype="float64"), xi), axis=1))
        adjusted_cumH = tf.maximum(adjusted_cumH, 1e-10)  # 确保为正数
        
        # Cox部分似然的三个组成部分
        # 1. 事件样本的线性预测子之和
        event_linear_pred = tf.reduce_sum(tf.gather(linear_pred, indices=smallidx))
        
        # 2. 调整后基线风险的对数之和
        log_hazards = tf.reduce_sum(tf.math.log(adjusted_hazards))
        
        # 3. 风险集的指数项
        exp_linear_pred = tf.math.exp(linear_pred)
        risk_set_term = tf.reduce_sum(tf.math.multiply(adjusted_cumH, exp_linear_pred))
        
        # 负对数似然
        neg_log_likelihood = tf.math.negative(event_linear_pred + log_hazards - risk_set_term)
        
        # L1惩罚项
        l1_eta = lambda1 * tf.math.reduce_sum(tf.math.abs(eta))
        l1_xi = lambda2 * tf.math.reduce_sum(tf.math.abs(xi))
        l1_beta = lambda_beta * tf.math.reduce_sum(tf.math.abs(current_beta_t))  # 新增的beta_t惩罚
        
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
        
        # 计算梯度 - 包含beta_t以实现真正的稀疏优化
        if lambda_beta > 0:
            gradients = tape.gradient(loss_value, [eta, xi, beta_t])
            optimizer.apply_gradients(zip(gradients, [eta, xi, beta_t]))
            
            # 对beta_t应用软阈值化以实现稀疏性
            # 软阈值化: sign(x) * max(0, |x| - threshold)
            # 使用适中的阈值以产生平衡的稀疏性
            threshold = lambda_beta * 0.05  # 调整阈值倍数以获得更平衡的稀疏性
            beta_t_sign = tf.sign(beta_t)
            beta_t_abs = tf.abs(beta_t)
            beta_t_thresholded = beta_t_sign * tf.maximum(0.0, beta_t_abs - threshold)
            beta_t.assign(beta_t_thresholded)
            
        else:
            gradients = tape.gradient(loss_value, [eta, xi])
            optimizer.apply_gradients(zip(gradients, [eta, xi]))
        
        # 记录损失
        loss_history.append(loss_value.numpy())
        
        # 每50步打印一次进度
        if verbose and (step + 1) % 50 == 0:
            print(f"步骤 {step + 1}/{nsteps}, 损失: {loss_value.numpy():.6f}")
        
        # 检查收敛
        if step > 10:
            recent_losses = loss_history[-10:]
            if max(recent_losses) - min(recent_losses) < tolerance:
                if verbose:
                    print(f"在步骤 {step + 1} 收敛")
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
        smallidx = tf.where(status_np == 2)[:, 0]  # 提取第一列索引
        
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