#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TransCox High-Dimensional Sparse Version - SCIENTIFIC FIX v2 (Safe Mode)
Method: L1 Regularized Loss + Hard Thresholding + Numerical Stability Clips
"""

import tensorflow as tf
import numpy as np

def TransCox_Sparse(CovData, cumH, hazards, status, estR, Xinn, 
                    lambda1, lambda2, lambda_beta, 
                    learning_rate=0.004, nsteps=200,
                    tolerance=1e-6, verbose=True,
                    threshold_c=0.5): 
    
    # --- 1. Data Dimensions & Theory ---
    n_samples = CovData.shape[0]
    n_features = CovData.shape[1]
    
    # Theoretical Threshold calculation
    theoretical_tau = threshold_c * np.sqrt(np.log(n_features) / n_samples)
    
    if verbose:
        print(f"TransCox Sparse (Safe Mode): L1 Loss + Hard Thresholding")
        print(f"Theoretical Tau: {theoretical_tau:.5f} (C={threshold_c})")
    
    # --- 2. TF Conversions ---
    dtype = tf.float64
    XiData_tf = tf.constant(np.ascontiguousarray(Xinn), dtype=dtype)
    ppData_tf = tf.constant(np.ascontiguousarray(CovData), dtype=dtype)
    cQ_tf = tf.constant(np.ascontiguousarray(cumH).reshape(-1), dtype=dtype)
    dq_tf = tf.constant(np.ascontiguousarray(hazards).reshape(-1), dtype=dtype)
    estR_tf = tf.constant(np.ascontiguousarray(estR).reshape(-1), dtype=dtype)
    
    status_np = np.ascontiguousarray(status).reshape(-1)
    event_code = 2 if np.max(status_np) > 1 else 1
    smallidx = tf.constant(np.where(status_np == event_code)[0], dtype=tf.int64)
    
    n_hazards = dq_tf.shape[0]
    eta = tf.Variable(tf.zeros(n_features, dtype=dtype), name="eta")
    xi = tf.Variable(tf.zeros(n_hazards, dtype=dtype), name="xi")
    
    use_beta_mode = (lambda_beta > 0)

    
    if use_beta_mode:
        beta_t = tf.Variable(estR_tf, name="beta_t")
    else:
        beta_t = tf.Variable(tf.zeros(1, dtype=dtype), name="dummy")

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    
    # --- 3. Optimization Graph (Safe Version) ---
    @tf.function(jit_compile=True) 
    def train_step():
        with tf.GradientTape() as tape:
            if use_beta_mode:
                current_beta = beta_t
                current_eta = beta_t - estR_tf 
            else:
                current_eta = eta
                current_beta = estR_tf + eta

            # Cox Partial Likelihood
            linear_pred = tf.linalg.matvec(ppData_tf, current_beta)
            xi_contribution = tf.linalg.matvec(XiData_tf, xi)
            
            # [SAFETY 1] Clip Linear Predictor to prevent exp() explosion
            # range [-15, 15] is sufficient for survival risk scores
            linear_pred = tf.clip_by_value(linear_pred, -15.0, 15.0)
            
            adjusted_cumH = tf.maximum(cQ_tf + xi_contribution, 1e-8)
            
            # [SAFETY 2] Ensure hazards are strictly positive for log()
            adjusted_hazards = tf.maximum(dq_tf + xi, 1e-8)
            
            event_linear_pred = tf.reduce_sum(tf.gather(linear_pred, smallidx))
            log_hazards = tf.reduce_sum(tf.math.log(adjusted_hazards))
            
            exp_linear_pred = tf.math.exp(linear_pred)
            risk_set_term = tf.reduce_sum(adjusted_cumH * exp_linear_pred)
            
            neg_log_likelihood = -(event_linear_pred + log_hazards - risk_set_term)
            
            # Regularization
            reg_xi = lambda2 * tf.reduce_sum(tf.abs(xi))
            
            if use_beta_mode:
                reg_eta = lambda1 * tf.reduce_sum(tf.abs(current_eta))
                reg_beta = lambda_beta * tf.reduce_sum(tf.abs(current_beta))
                loss = neg_log_likelihood + reg_eta + reg_xi + reg_beta
            else:
                reg_eta = lambda1 * tf.reduce_sum(tf.abs(eta))
                loss = neg_log_likelihood + reg_eta + reg_xi

        # Update
        if use_beta_mode:
            vars_to_update = [beta_t, xi]
        else:
            vars_to_update = [eta, xi]
            
        gradients = tape.gradient(loss, vars_to_update)
        
        # [SAFETY 3] Clip Gradients to prevent NaN propagation
        # If a single sample causes a spike, this saves the model.
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
        
        optimizer.apply_gradients(zip(gradients, vars_to_update))
        return loss

    # --- 4. Execution ---
    loss_history = []
    n_steps_int = int(nsteps)
    
    for step in range(n_steps_int):
        loss_val = train_step()
        loss_val_float = float(loss_val)
        
        # Check for NaN immediately
        if np.isnan(loss_val_float):
            if verbose: print(f"Warning: Loss is NaN at step {step}. Stopping.")
            break
            
        loss_history.append(loss_val_float)
        
        if verbose and (step + 1) % 500 == 0:
             print(f"Step {step + 1}/{n_steps_int}, Loss: {loss_val_float:.4f}")

    # --- 5. Post-Hoc Hard Thresholding ---
    eta_final = eta.numpy()
    xi_final = xi.numpy()
    
    if use_beta_mode:
        beta_raw = beta_t.numpy()
    else:
        beta_raw = estR + eta_final
        
    # Handle NaN in output (Fallback to EstR if training failed)
    if np.any(np.isnan(beta_raw)):
        if verbose: print("Error: Training produced NaNs. Falling back to source estimates.")
        beta_raw = np.nan_to_num(estR_tf.numpy()) # Fallback
    
    # Apply Theoretical Threshold
    mask = np.abs(beta_raw) >= theoretical_tau
    beta_final = beta_raw * mask
    
    eta_final_corrected = beta_final - estR
    nonzero_beta = np.sum(np.abs(beta_final) > 1e-8)
    
    convergence_info = {
        'loss_history': loss_history,
        'theoretical_tau': theoretical_tau,
        'nonzero_beta': nonzero_beta
    }
    
    return eta_final_corrected, xi_final, beta_final, convergence_info