#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TransCox High-Dimensional Sparse - FINAL FUSION VERSION
Method: Proximal Gradient Descent (PGD) with Gradient Clipping
Logic: 
  1. Matches JASA Original Term 1 Logic (Vector Sum).
  2. Adds Gradient Clipping (Critical for stability).
"""

import tensorflow as tf
import numpy as np

def TransCox_Sparse(CovData, cumH, hazards, status, estR, Xinn, 
                    lambda1, lambda2, lambda_beta, 
                    learning_rate=0.001, # [Config] Lower LR for high-dim stability
                    nsteps=3000,
                    tolerance=1e-7, verbose=True):
    
    # --- 1. Data Setup ---
    dtype = tf.float64
    
    X_tf = tf.constant(np.ascontiguousarray(CovData), dtype=dtype)
    Xi_tf = tf.constant(np.ascontiguousarray(Xinn), dtype=dtype)
    
    # cQ_tf: Source Cumulative Hazard H0(t)
    cQ_tf = tf.constant(np.ascontiguousarray(cumH).reshape(-1), dtype=dtype)
    
    # dq_tf: Source Instantaneous Hazard h0(t)
    dq_tf = tf.constant(np.ascontiguousarray(hazards).reshape(-1), dtype=dtype)
    
    estR_tf = tf.constant(np.ascontiguousarray(estR).reshape(-1), dtype=dtype)
    
    status_np = np.ascontiguousarray(status).reshape(-1)
    event_code = 2 if np.max(status_np) > 1 else 1
    event_indices = tf.constant(np.where(status_np == event_code)[0], dtype=tf.int64)
    n_events = tf.cast(tf.shape(event_indices)[0], dtype=dtype)
    
    # xi dimension matches dq (Instantaneous hazard vector)
    n_hazards = dq_tf.shape[0] 
    
    # Initialize Variables
    beta_t = tf.Variable(estR_tf, name="beta_t") 
    xi = tf.Variable(tf.zeros(n_hazards, dtype=dtype), name="xi")

    # --- 2. Proximal Operators ---
    @tf.function
    def soft_threshold(x, lam):
        return tf.math.sign(x) * tf.maximum(tf.math.abs(x) - lam, 0.0)

    @tf.function
    def proximal_double_l1(u, center, lam_transfer, lam_sparsity):
        diff = u - center
        z = soft_threshold(diff, lam_transfer) + center
        final = soft_threshold(z, lam_sparsity)
        return final

    # --- 3. Gradient Calculation ---
    @tf.function
    def compute_gradients_and_loss():
        with tf.GradientTape() as tape:
            
            # === PART A: Term 1 (Event Log-Likelihood) ===
            # Original JASA Logic: Sum of log hazards for unique times.
            # Assuming 'dq_tf' (and thus 'xi') corresponds to the events.
            
            h_new = dq_tf + xi
            h_new_safe = tf.maximum(h_new, 1e-10) # Safety
            
            # [LOGIC MATCH] Use simple reduce_sum matching original author code
            term1_h = tf.reduce_sum(tf.math.log(h_new_safe))
            
            # Linear Predictor Contribution: sum(X * beta) for events
            linear_pred = tf.linalg.matvec(X_tf, beta_t)
            # [SAFETY] Clip linear predictor before exp
            linear_pred_safe = tf.clip_by_value(linear_pred, -15.0, 15.0)
            
            term1_beta = tf.reduce_sum(tf.gather(linear_pred_safe, event_indices))
            
            term1 = term1_beta + term1_h
            
            # === PART B: Term 2 (Risk Set Integral) ===
            # Original: sum((cQ + sum(XiData * xi)) * exp(XB))
            
            # 1. Cumulative xi: Sum_{j <= i} xi_j
            cum_xi = tf.linalg.matvec(Xi_tf, xi)
            
            # 2. Total Cumulative Hazard H(t)
            H_new = cQ_tf + cum_xi
            H_new_safe = tf.maximum(H_new, 1e-10)
            
            # 3. Risk Score
            risk_score = tf.math.exp(linear_pred_safe)
            
            # Sum over all subjects
            term2 = tf.reduce_sum(H_new_safe * risk_score)
            
            # === Loss ===
            nll = -(term1 - term2)
            scaled_nll = nll / n_events

        grads = tape.gradient(scaled_nll, [beta_t, xi])
        
        # [CRITICAL ADDITION] Gradient Clipping
        # This prevents the coefficients from exploding to 30,000
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        
        return grads, scaled_nll

    # --- 4. Training Loop ---
    lr = tf.constant(learning_rate, dtype=dtype)
    loss_history = []
    
    for step in range(nsteps):
        (d_beta, d_xi), current_loss = compute_gradients_and_loss()
        
        if tf.math.is_nan(current_loss):
            if verbose: print(f"Warning: NaN Loss at step {step}")
            break
            
        u_beta = beta_t - lr * d_beta
        new_beta = proximal_double_l1(u_beta, estR_tf, lambda1 * lr, lambda_beta * lr)
        beta_t.assign(new_beta)
        
        u_xi = xi - lr * d_xi
        new_xi = soft_threshold(u_xi, lambda2 * lr)
        xi.assign(new_xi)
        
        loss_history.append(float(current_loss))

    # --- 5. Output ---
    beta_final = beta_t.numpy()
    xi_final = xi.numpy()
    eta_final = beta_final - estR
    
    nonzero_beta = np.sum(np.abs(beta_final) > 1e-8)
    
    convergence_info = {
        'loss_history': loss_history,
        'nonzero_beta': nonzero_beta,
        'final_loss': loss_history[-1] if loss_history else 0
    }
    
    return eta_final, xi_final, beta_final, convergence_info