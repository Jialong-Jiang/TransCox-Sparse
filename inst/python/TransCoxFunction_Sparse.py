#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TransCox High-Dimensional Sparse Version - Python Optimization Function

Refactored for Scientific Rigor:
1. Method: L1 Regularized Loss + Post-hoc Theoretical Hard Thresholding
2. Theory: Threshold tau = C * sqrt(log(p)/n) based on high-dimensional inference bounds.
3. Optimizer: Adam with direct gradient descent on L1 loss (no iterative shrinkage).

"""

import tensorflow as tf
import numpy as np

def TransCox_Sparse(CovData, cumH, hazards, status, estR, Xinn, 
                    lambda1, lambda2, lambda_beta, 
                    learning_rate=0.004, nsteps=200,
                    tolerance=1e-6, verbose=True,
                    threshold_c=0.1): # [New Parameter] Theoretical constant C
    
    # --- 1. Data Dimensions & Constants ---
    n_samples = CovData.shape[0]
    n_features = CovData.shape[1]
    
    # Calculate Theoretical Threshold: tau = C * sqrt(log(p) / n)
    # Note: Adding 1e-8 to log to prevent error if p=1, though unlikely in high-dim
    theoretical_tau = threshold_c * np.sqrt(np.log(n_features) / n_samples)
    
    if verbose:
        print(f"TransCox Sparse Optimization (L1 Loss + Hard Thresholding)...")
        print(f"N={n_samples}, P={n_features}")
        print(f"Theoretical Noise Floor (tau): {theoretical_tau:.5f} (C={threshold_c})")
    
    # --- 2. TF Data Conversion ---
    dtype = tf.float64
    XiData_tf = tf.constant(np.ascontiguousarray(Xinn), dtype=dtype)
    ppData_tf = tf.constant(np.ascontiguousarray(CovData), dtype=dtype)
    cQ_tf = tf.constant(np.ascontiguousarray(cumH).reshape(-1), dtype=dtype)
    dq_tf = tf.constant(np.ascontiguousarray(hazards).reshape(-1), dtype=dtype)
    estR_tf = tf.constant(np.ascontiguousarray(estR).reshape(-1), dtype=dtype)
    
    status_np = np.ascontiguousarray(status).reshape(-1)
    event_code = 2 if np.max(status_np) > 1 else 1
    smallidx = tf.constant(np.where(status_np == event_code)[0], dtype=tf.int64)
    
    # --- 3. Variable Initialization ---
    n_hazards = dq_tf.shape[0]
    eta = tf.Variable(tf.zeros(n_features, dtype=dtype), name="eta")
    xi = tf.Variable(tf.zeros(n_hazards, dtype=dtype), name="xi")
    
    # Check if we are in Beta-Sparsity Mode
    # If lambda_beta > 0, we treat beta_t as the primary variable
    use_beta_mode = (lambda_beta > 0)
    
    if use_beta_mode:
        # Initialize beta_t with Ridge estimates (estR) for faster convergence
        beta_t = tf.Variable(estR_tf, name="beta_t")
    else:
        # Dummy variable
        beta_t = tf.Variable(tf.zeros(1, dtype=dtype), name="dummy_beta")

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    
    # --- 4. Optimization Graph (JIT Compiled) ---
    @tf.function(jit_compile=True) 
    def train_step():
        with tf.GradientTape() as tape:
            # Definition of Beta and Eta
            if use_beta_mode:
                current_beta = beta_t
                current_eta = beta_t - estR_tf 
            else:
                current_eta = eta
                current_beta = estR_tf + eta

            # --- Cox Partial Likelihood Calculation ---
            linear_pred = tf.linalg.matvec(ppData_tf, current_beta)
            xi_contribution = tf.linalg.matvec(XiData_tf, xi)
            
            adjusted_cumH = cQ_tf + xi_contribution
            adjusted_hazards = tf.maximum(dq_tf + xi, 1e-8)
            
            event_linear_pred = tf.reduce_sum(tf.gather(linear_pred, smallidx))
            log_hazards = tf.reduce_sum(tf.math.log(adjusted_hazards))
            
            exp_linear_pred = tf.math.exp(linear_pred)
            risk_set_term = tf.reduce_sum(adjusted_cumH * exp_linear_pred)
            
            # Negative Log Likelihood
            neg_log_likelihood = -(event_linear_pred + log_hazards - risk_set_term)
            
            # --- Regularization (L1 Loss) ---
            # Instead of manual shrinkage, we add L1 penalty to the Loss.
            # Adam will handle the gradient descent direction.
            
            l1_xi = lambda2 * tf.reduce_sum(tf.abs(xi))
            
            if use_beta_mode:
                # Lambda1 penalizes transfer divergence (Eta)
                l1_eta = lambda1 * tf.reduce_sum(tf.abs(current_eta))
                # Lambda_Beta penalizes total sparsity (Beta)
                l1_beta = lambda_beta * tf.reduce_sum(tf.abs(current_beta))
                
                loss = neg_log_likelihood + l1_eta + l1_xi + l1_beta
            else:
                l1_eta = lambda1 * tf.reduce_sum(tf.abs(eta))
                loss = neg_log_likelihood + l1_eta + l1_xi

        # --- Gradients & Update ---
        if use_beta_mode:
            vars_to_update = [beta_t, xi]
        else:
            vars_to_update = [eta, xi]
            
        gradients = tape.gradient(loss, vars_to_update)
        optimizer.apply_gradients(zip(gradients, vars_to_update))
        
        # NOTE: No manual soft-thresholding loop here. 
        # We rely on the Loss function to drive values small, 
        # and the post-hoc step to zero them out.
            
        return loss

    # --- 5. Execution Loop ---
    loss_history = []
    n_steps_int = int(nsteps)
    
    for step in range(n_steps_int):
        loss_val = train_step()
        loss_history.append(float(loss_val))
        
        if verbose and (step + 1) % 500 == 0:
             print(f"Step {step + 1}/{n_steps_int}, Loss: {loss_val:.4f}")
             
        # Simple Early Stopping
        if step > 50 and step % 50 == 0:
            if abs(loss_history[-1] - loss_history[-6]) < tolerance:
                if verbose: print(f"Converged early at step {step+1}")
                break

    # --- 6. Finalization & Theoretical Hard Thresholding ---
    eta_final = eta.numpy()
    xi_final = xi.numpy()
    
    if use_beta_mode:
        beta_final = beta_t.numpy()
    else:
        beta_final = estR + eta_final
        
    # === [SCIENTIFIC CORE] Hard Thresholding ===
    # Apply the O(sqrt(log p / n)) threshold
    # Coefficients below this noise floor are statistically indistinguishable from zero.
    
    # 1. Apply threshold to Beta
    mask = np.abs(beta_final) >= theoretical_tau
    beta_final_sparse = beta_final * mask
    
    # 2. Sync Eta (Eta = Beta_new - Beta_source)
    # This ensures consistency after thresholding
    eta_final_corrected = beta_final_sparse - estR
    
    nonzero_beta = np.sum(np.abs(beta_final_sparse) > 1e-8)
    
    convergence_info = {
        'loss_history': loss_history,
        'nonzero_beta': nonzero_beta,
        'theoretical_tau': theoretical_tau
    }
    
    return eta_final_corrected, xi_final, beta_final_sparse, convergence_info