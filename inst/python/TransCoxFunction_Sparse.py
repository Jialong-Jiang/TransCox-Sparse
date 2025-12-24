#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TransCox High-Dimensional Sparse Version - Python Optimization Function

Extends the original TransCox to support high-dimensional sparse data:
1. Adds lambda_beta * ||beta_t||_1 penalty term
2. Supports sparse coefficient output
3. Improved optimization algorithm

"""

import tensorflow as tf
import numpy as np

def TransCox_Sparse(CovData, cumH, hazards, status, estR, Xinn, 
                    lambda1, lambda2, lambda_beta=[0.01,0.05,0.1,0.2,0.5,1,2],
                    learning_rate=0.004, nsteps=200,
                    tolerance=1e-6, verbose=True):
    
    if verbose:
        print(f"TransCox Sparse Optimization (Accelerated Version)...")
        print(f"Samples: {CovData.shape[0]}, Features: {CovData.shape[1]}")
        print(f"LR: {learning_rate}, Lambda_Beta: {lambda_beta}")
    
    # --- Data Preprocessing (Convert to TF Constants once) ---
    # Using float64 for stability as requested
    dtype = tf.float64
    
    XiData_tf = tf.constant(np.ascontiguousarray(Xinn), dtype=dtype)
    ppData_tf = tf.constant(np.ascontiguousarray(CovData), dtype=dtype)
    cQ_tf = tf.constant(np.ascontiguousarray(cumH).reshape(-1), dtype=dtype)
    dq_tf = tf.constant(np.ascontiguousarray(hazards).reshape(-1), dtype=dtype)
    estR_tf = tf.constant(np.ascontiguousarray(estR).reshape(-1), dtype=dtype)
    
    status_np = np.ascontiguousarray(status).reshape(-1)
    event_code = 2 if np.max(status_np) > 1 else 1
    smallidx = tf.constant(np.where(status_np == event_code)[0], dtype=tf.int64)
    
    # [Optimized] Pre-calculate N_events for potential use, though we reverted standardization
    # keeping it logical.
    
    # --- Variables ---
    # Initialize variables
    n_features = ppData_tf.shape[1]
    n_hazards = dq_tf.shape[0]
    
    eta = tf.Variable(tf.zeros(n_features, dtype=dtype), name="eta")
    xi = tf.Variable(tf.zeros(n_hazards, dtype=dtype), name="xi")
    
    use_beta_mode = (lambda_beta > 0)
    
    if use_beta_mode:
        beta_t = tf.Variable(estR_tf, name="beta_t")
    else:
        # Dummy variable to satisfy graph requirements if not used
        beta_t = tf.Variable(tf.zeros(1, dtype=dtype), name="dummy_beta")

    # Optimizer
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    
    # --- Define Graph Function (The Speed Secret) ---
    # jit_compile=True enables XLA optimization (maximum speed)
    @tf.function(jit_compile=True) 
    def train_step():
        with tf.GradientTape() as tape:
            if use_beta_mode:
                current_beta_t = beta_t
                current_eta = beta_t - estR_tf 
            else:
                current_eta = eta
                current_beta_t = estR_tf + eta

            # Matrix Operations
            linear_pred = tf.linalg.matvec(ppData_tf, current_beta_t)
            xi_contribution = tf.linalg.matvec(XiData_tf, xi)
            
            # Cox Partial Likelihood Components
            adjusted_cumH = cQ_tf + xi_contribution
            adjusted_hazards = tf.maximum(dq_tf + xi, 1e-8) # Stability
            
            event_linear_pred = tf.reduce_sum(tf.gather(linear_pred, smallidx))
            log_hazards = tf.reduce_sum(tf.math.log(adjusted_hazards))
            
            exp_linear_pred = tf.math.exp(linear_pred)
            risk_set_term = tf.reduce_sum(adjusted_cumH * exp_linear_pred)
            
            # Loss Calculation (Sum, Unscaled)
            neg_log_likelihood = -(event_linear_pred + log_hazards - risk_set_term)
            
            l1_xi = lambda2 * tf.reduce_sum(tf.abs(xi))
            
            if use_beta_mode:
                l1_eta = lambda1 * tf.reduce_sum(tf.abs(current_eta))
                loss = neg_log_likelihood + l1_eta + l1_xi
            else:
                l1_eta = lambda1 * tf.reduce_sum(tf.abs(eta))
                loss = neg_log_likelihood + l1_eta + l1_xi

        # --- Gradients & Updates ---
        if use_beta_mode:
            vars_to_update = [beta_t, xi]
            gradients = tape.gradient(loss, vars_to_update)
            optimizer.apply_gradients(zip(gradients, vars_to_update))
            
            # Proximal Operator (Soft Thresholding)
            # Threshold = LR * Lambda
            threshold = learning_rate * lambda_beta
            
            # Vectorized soft thresholding
            beta_sign = tf.sign(beta_t)
            beta_abs = tf.abs(beta_t)
            beta_new = beta_sign * tf.maximum(beta_abs - threshold, 0.0)
            beta_t.assign(beta_new)
            
            # Sync eta
            eta.assign(beta_t - estR_tf)
            
        else:
            vars_to_update = [eta, xi]
            gradients = tape.gradient(loss, vars_to_update)
            optimizer.apply_gradients(zip(gradients, vars_to_update))
            
            # Soft threshold eta
            eta_thresh = learning_rate * lambda1 * 0.1
            eta.assign(tf.sign(eta) * tf.maximum(tf.abs(eta) - eta_thresh, 0.0))
            
            # Soft threshold xi
            xi_thresh = learning_rate * lambda2 * 0.1
            xi.assign(tf.sign(xi) * tf.maximum(tf.abs(xi) - xi_thresh, 0.0))
            
        return loss

    # --- Execution Loop ---
    loss_history = []
    
    # Convert nsteps to int
    n_steps_int = int(nsteps)
    
    for step in range(n_steps_int):
        # Run the compiled graph step
        loss_val = train_step()
        
        # Record loss (must convert tensor to numpy float)
        current_loss = float(loss_val)
        loss_history.append(current_loss)
        
        # Logging
        if verbose and (step + 1) % 500 == 0:
             print(f"Step {step + 1}/{n_steps_int}, Loss: {current_loss:.4f}")
             
        # Early Stopping Check (Every 50 steps to save overhead)
        if step > 50 and step % 50 == 0:
            # Check if loss change over last 5 steps is tiny
            if abs(loss_history[-1] - loss_history[-6]) < tolerance:
                if verbose:
                    print(f"Converged early at step {step+1}")
                break

    # --- Finalizing Results ---
    eta_final = eta.numpy()
    xi_final = xi.numpy()
    
    if use_beta_mode:
        beta_t_final = beta_t.numpy()
    else:
        beta_t_final = estR_np + eta_final
    
    nonzero_beta = np.sum(np.abs(beta_t_final) > 1e-6)
    
    convergence_info = {
        'loss_history': loss_history,
        'nonzero_beta': nonzero_beta,
        'steps_taken': len(loss_history)
    }
    
    return eta_final, xi_final, beta_t_final, convergence_info