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
import tensorflow_probability as tfp
import numpy as np

def TransCox_Sparse(CovData, cumH, hazards, status, estR, Xinn, 
                    lambda1, lambda2, lambda_beta=[0.01, 0.025,0.05,0.1,0.2,0.5],
                    learning_rate=0.004, nsteps=200,
                    tolerance=1e-6, verbose=True):
    
    if verbose:
        print(f"TransCox Sparse Optimization (Fixed Version)...")
        print(f"Samples: {CovData.shape[0]}, Features: {CovData.shape[1]}")
    
    # --- Data Preprocessing ---
    XiData = np.ascontiguousarray(Xinn, dtype=np.float64)
    ppData = np.ascontiguousarray(CovData, dtype=np.float64)
    cQ_np = np.ascontiguousarray(cumH, dtype=np.float64).reshape((len(cumH),))
    dq_np = np.ascontiguousarray(hazards, dtype=np.float64).reshape((len(hazards),))
    estR_np = np.ascontiguousarray(estR, dtype=np.float64).reshape((len(estR),))
    status_np = np.ascontiguousarray(status, dtype=np.float64).reshape((len(status),))
    
    event_code = 2 if np.max(status_np) > 1 else 1
    event_mask = status_np == event_code
    smallidx = tf.constant(np.where(event_mask)[0], dtype=tf.int64)
    
    # [FIX 1] Capture number of events for scaling
    n_events_num = tf.cast(len(smallidx), tf.float64) 
    
    # --- Variables ---
    eta = tf.Variable(np.zeros(ppData.shape[1], dtype=np.float64), dtype=tf.float64, name="eta")
    xi = tf.Variable(np.zeros(len(dq_np), dtype=np.float64), dtype=tf.float64, name="xi")
    
    # If lambda_beta > 0, we optimize beta_t directly
    use_beta_mode = (lambda_beta > 0)
    if use_beta_mode:
        beta_t = tf.Variable(estR_np.copy(), dtype=tf.float64, name="beta_t")
    
    # Constants
    ppData_tf = tf.constant(ppData, dtype=tf.float64)
    cQ_tf = tf.constant(cQ_np, dtype=tf.float64)
    dq_tf = tf.constant(dq_np, dtype=tf.float64)
    estR_tf = tf.constant(estR_np, dtype=tf.float64)
    XiData_tf = tf.constant(XiData, dtype=tf.float64)

    # Optimizer
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    loss_history = []
    
    # --- Soft Thresholding Function ---
    def soft_threshold(x, threshold):
        return tf.sign(x) * tf.maximum(tf.abs(x) - threshold, 0.0)

    # --- Optimization Loop ---
    for step in range(int(nsteps)):
        with tf.GradientTape() as tape:
            # 1. Determine Current Beta
            if use_beta_mode:
                current_beta_t = beta_t
                # Note: eta is derived for loss calculation
                current_eta = beta_t - estR_tf 
            else:
                current_eta = eta
                current_beta_t = estR_tf + eta

            # 2. Cox Partial Likelihood Calculation
            linear_pred = tf.linalg.matvec(ppData_tf, current_beta_t)
            xi_contribution = tf.linalg.matvec(XiData_tf, xi)
            adjusted_cumH = cQ_tf + xi_contribution
            adjusted_hazards = tf.maximum(dq_tf + xi, 1e-8)
            
            event_linear_pred = tf.reduce_sum(tf.gather(linear_pred, smallidx))
            log_hazards = tf.reduce_sum(tf.math.log(adjusted_hazards))
            
            exp_linear_pred = tf.math.exp(linear_pred)
            risk_set_term = tf.reduce_sum(adjusted_cumH * exp_linear_pred)
            
            neg_log_likelihood = -(event_linear_pred + log_hazards - risk_set_term)
            
            # [FIX 1] Scale Likelihood by Number of Events (Normalize)
            # This makes lambda invariant to sample size
            scaled_nll = neg_log_likelihood / n_events_num

            # 3. Penalties (For Smooth Gradient Part)
            # We only include penalties that we want to optimize via Gradient Descent (sub-gradient)
            # We EXCLUDE l1_beta here because we will handle it with Proximal Operator later
            
            l1_xi = lambda2 * tf.reduce_sum(tf.abs(xi))
            
            if use_beta_mode:
                # We keep Transfer Loss (l1_eta) in the gradient part
                # Because it centers around estR, not 0.
                l1_eta = lambda1 * tf.reduce_sum(tf.abs(current_eta))
                loss_for_tape = scaled_nll + l1_eta + l1_xi
                # NOTE: l1_beta is NOT added here!
            else:
                l1_eta = lambda1 * tf.reduce_sum(tf.abs(eta))
                loss_for_tape = scaled_nll + l1_eta + l1_xi

        # --- Gradient Update ---
        if use_beta_mode:
            gradients = tape.gradient(loss_for_tape, [beta_t, xi])
            optimizer.apply_gradients(zip(gradients, [beta_t, xi]))
            
            # [FIX 2 & 3] Proximal Operator (Correct Soft Thresholding)
            # The threshold must be: Learning Rate * Lambda
            # Since we scaled NLL by 1/N, this lambda is now "per event" scale.
            
            # Effective threshold considering Adam's adaptive rate is complex, 
            # but for sparsity, "lr * lambda" is the standard approximation.
            prox_threshold = learning_rate * lambda_beta
            beta_t.assign(soft_threshold(beta_t, prox_threshold))
            
            # Update eta for consistency
            eta.assign(beta_t - estR_tf)
            
        else:
            gradients = tape.gradient(loss_for_tape, [eta, xi])
            optimizer.apply_gradients(zip(gradients, [eta, xi]))
            
            # Soft threshold for eta/xi (optional, for clean 0s)
            eta.assign(soft_threshold(eta, learning_rate * lambda1 * 0.1))
            xi.assign(soft_threshold(xi, learning_rate * lambda2 * 0.1))

        loss_history.append(loss_for_tape.numpy())
        
        # Verbose & Convergence (Simplified for brevity)
        if verbose and (step + 1) % 50 == 0:
             print(f"Step {step + 1}, Loss: {loss_for_tape.numpy():.6f}")

    # --- Final Output Preparation ---
    eta_final = eta.numpy()
    xi_final = xi.numpy()
    beta_t_final = beta_t.numpy() if use_beta_mode else (estR_np + eta_final)
    
    # Calculate sparsity stats
    nonzero_beta = np.sum(np.abs(beta_t_final) > 1e-6)
    
    convergence_info = {
        'loss_history': loss_history,
        'nonzero_beta': nonzero_beta
    }
    
    return eta_final, xi_final, beta_t_final, convergence_info

def TransCox(CovData, cumH, hazards, status, estR, Xinn, 
            lambda1, lambda2, learning_rate=0.004, nsteps=200,
            lambda_beta=None):
    """
    Backward-compatible TransCox function
    
    If lambda_beta is not specified or 0, use original algorithm
    Otherwise use new sparse algorithm
    """
    
    if lambda_beta is None or lambda_beta == 0:
        # Use original algorithm
        xi = tf.Variable(np.repeat([0.], repeats=len(hazards)), dtype="float64")
        eta = tf.Variable(np.repeat([0.], repeats=len(estR)), dtype="float64")
        
        XiData = np.float64(Xinn)
        ppData = np.float64(CovData)
        cQ_np = np.float64(cumH).reshape((len(cumH),))
        dq_np = np.float64(hazards).reshape((len(hazards),))
        estR2 = np.float64(estR).reshape((len(estR),))
        status_np = np.float64(status).reshape((len(status),))
        # Compatible with 0/1 and 1/2 status encoding
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
        # Use new sparse algorithm
        eta, xi, beta_t, conv_info = TransCox_Sparse(
            CovData, cumH, hazards, status, estR, Xinn,
            lambda1, lambda2, lambda_beta, learning_rate, nsteps, verbose=False
        )
        
        return eta, xi