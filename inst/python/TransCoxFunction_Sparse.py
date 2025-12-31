#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TransCox High-Dimensional Sparse - JASA ADDITIVE VERSION
Method: Proximal Gradient Descent (PGD) with Additive Hazard Transfer
Status: Scientific & Numerically Safe
"""

import tensorflow as tf
import numpy as np

def TransCox_Sparse(CovData, cumH, hazards, status, estR, Xinn, 
                    lambda1, lambda2, lambda_beta, 
                    learning_rate=0.01, nsteps=2000,
                    tolerance=1e-7, verbose=True):
    
    # --- 1. Data Setup ---
    dtype = tf.float64
    
    X_tf = tf.constant(np.ascontiguousarray(CovData), dtype=dtype)
    Xi_tf = tf.constant(np.ascontiguousarray(Xinn), dtype=dtype)
    
    # Baseline Hazard Quantities (Source)
    cQ_tf = tf.constant(np.ascontiguousarray(cumH).reshape(-1), dtype=dtype)   # H0_source
    dq_tf = tf.constant(np.ascontiguousarray(hazards).reshape(-1), dtype=dtype) # h0_source
    estR_tf = tf.constant(np.ascontiguousarray(estR).reshape(-1), dtype=dtype) # beta_source
    
    status_np = np.ascontiguousarray(status).reshape(-1)
    event_code = 2 if np.max(status_np) > 1 else 1
    event_indices = tf.constant(np.where(status_np == event_code)[0], dtype=tf.int64)
    n_events = tf.cast(tf.shape(event_indices)[0], dtype=dtype)
    
    n_features = X_tf.shape[1]
    n_hazards = dq_tf.shape[0]

    # Initialize Variables
    # We optimize Beta_Target directly.
    beta_t = tf.Variable(estR_tf, name="beta_t") 
    xi = tf.Variable(tf.zeros(n_hazards, dtype=dtype), name="xi")

    # --- 2. The Proximal Operators ---
    
    @tf.function(jit_compile=True)
    def soft_threshold(x, lam):
        return tf.math.sign(x) * tf.maximum(tf.math.abs(x) - lam, 0.0)

    @tf.function(jit_compile=True)
    def proximal_double_l1(u, center, lam_transfer, lam_sparsity):
        """
        Sequential Shrinkage for Beta:
        1. Shrink towards Source (Transfer)
        2. Shrink towards 0 (Sparsity)
        """
        # Step 1: Transfer (eta penalty)
        diff = u - center
        z = soft_threshold(diff, lam_transfer) + center
        
        # Step 2: Sparsity (beta penalty)
        final = soft_threshold(z, lam_sparsity)
        return final

    # --- 3. Gradient Calculation (Additive Model) ---
    @tf.function(jit_compile=True)
    def compute_gradients_and_loss():
        with tf.GradientTape() as tape:

            xi_contribution = tf.linalg.matvec(Xi_tf, xi) 
            


            h0_source_mapped = tf.linalg.matvec(Xi_tf, dq_tf) # Map unique h0 to individuals
            
            h_new_mapped = h0_source_mapped + xi_contribution
            
            # [SAFETY] Clip Hazard > 1e-10
            h_new_safe = tf.maximum(h_new_mapped, 1e-10)
            
            # Linear Predictor
            linear_pred = tf.linalg.matvec(X_tf, beta_t)
            linear_pred = tf.clip_by_value(linear_pred, -15.0, 15.0)
            
            # Term 1: Event Log-Likelihood
            # Sum_{events} [ X*beta + log(h0 + xi) ]
            lp_events = tf.gather(linear_pred, event_indices)
            log_h_events = tf.math.log(tf.gather(h_new_safe, event_indices))
            
            term1 = tf.reduce_sum(lp_events + log_h_events)
            
           
            risk_score = tf.math.exp(linear_pred)
            

            cum_xi = tf.linalg.matvec(Xi_tf, xi)
            H_new_safe = tf.maximum(cQ_tf + cum_xi, 1e-10)
            
            term2 = tf.reduce_sum(H_new_safe * risk_score)
            
            # NLL
            nll = -(term1 - term2)
            scaled_nll = nll / n_events

        grads = tape.gradient(scaled_nll, [beta_t, xi])
        return grads, scaled_nll

    # --- 4. Training Loop (PGD) ---
    lr = tf.constant(learning_rate, dtype=dtype)
    loss_history = []
    
    for step in range(nsteps):
        
        # 4.1 Gradient
        (d_beta, d_xi), current_loss = compute_gradients_and_loss()
        
        if tf.math.is_nan(current_loss):
            if verbose: print(f"Warning: NaN Loss at step {step}")
            break
            
        # 4.2 Beta Update (Double Shrinkage)
        u_beta = beta_t - lr * d_beta
        new_beta = proximal_double_l1(
            u_beta, 
            center=estR_tf, 
            lam_transfer=lambda1 * lr, 
            lam_sparsity=lambda_beta * lr
        )
        beta_t.assign(new_beta)
        
        # 4.3 Xi Update (Soft Threshold for L1 on Xi)
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