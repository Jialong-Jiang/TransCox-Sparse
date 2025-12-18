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
                   lambda1, lambda2, lambda_beta=0.0,
                   learning_rate=0.004, nsteps=200,
                   tolerance=1e-6, verbose=True):
    """
    High-Dimensional Sparse TransCox Optimization Function
    
    Parameters:
    --------
    CovData : array-like, shape (n_samples, n_features)
        Covariate matrix
    cumH : array-like, shape (n_samples,)
        Cumulative baseline hazards
    hazards : array-like, shape (n_events,)
        Baseline hazard increments
    status : array-like, shape (n_samples,)
        Survival status indicator
    estR : array-like, shape (n_features,)
        Estimated coefficients beta_s from source domain
    Xinn : array-like, shape (n_samples, n_events)
        Event indicator matrix
    lambda1 : float
        L1 penalty parameter for eta (difference)
    lambda2 : float
        L1 penalty parameter for xi (baseline adjustment)
    lambda_beta : float
        L1 penalty parameter for beta_t = estR + eta (new)
    learning_rate : float
        Learning rate
    nsteps : int
        Number of optimization steps
    tolerance : float
        Convergence tolerance
    verbose : bool
        Whether to print detailed information
        
    Returns:
    --------
    tuple : (eta, xi, beta_t, convergence_info)
        eta : Coefficient difference
        xi : Baseline hazard adjustment
        beta_t : Final coefficients (estR + eta)
        convergence_info : Convergence information
    """
    
    if verbose:
        print(f"TransCox Sparse Optimization Started...")
        print(f"Samples: {CovData.shape[0]}, Features: {CovData.shape[1]}")
        print(f"lambda1 (eta): {lambda1}, lambda2 (xi): {lambda2}, lambda_beta: {lambda_beta}")
    
    # Optimized data preprocessing: reduce memory allocation
    XiData = np.ascontiguousarray(Xinn, dtype=np.float64)
    ppData = np.ascontiguousarray(CovData, dtype=np.float64)
    cQ_np = np.ascontiguousarray(cumH, dtype=np.float64).reshape((len(cumH),))
    dq_np = np.ascontiguousarray(hazards, dtype=np.float64).reshape((len(hazards),))
    estR_np = np.ascontiguousarray(estR, dtype=np.float64).reshape((len(estR),))
    status_np = np.ascontiguousarray(status, dtype=np.float64).reshape((len(status),))
    
    # Pre-calculate common values
    n_samples, n_features = ppData.shape
    n_events = len(dq_np)
    
    # Compatible with 0/1 and 1/2 status encoding
    event_code = 2 if np.max(status_np) > 1 else 1
    event_mask = status_np == event_code
    smallidx = tf.constant(np.where(event_mask)[0], dtype=tf.int64)
    n_events_actual = len(smallidx)
    
    # Optimized parameter initialization
    eta = tf.Variable(np.zeros(n_features, dtype=np.float64), dtype=tf.float64, name="eta")
    xi = tf.Variable(np.zeros(n_events, dtype=np.float64), dtype=tf.float64, name="xi")
    
    # When lambda_beta > 0, treat beta_t as an independent trainable variable
    if lambda_beta > 0:
        beta_t = tf.Variable(estR_np.copy(), dtype=tf.float64, name="beta_t")
    
    # Pre-calculate TensorFlow constants to avoid repeated conversion
    ppData_tf = tf.constant(ppData, dtype=tf.float64)
    cQ_tf = tf.constant(cQ_np, dtype=tf.float64)
    dq_tf = tf.constant(dq_np, dtype=tf.float64)
    estR_tf = tf.constant(estR_np, dtype=tf.float64)
    XiData_tf = tf.constant(XiData, dtype=tf.float64)
    
    # Loss history and convergence monitoring
    loss_history = []
    convergence_window = 5  # Convergence check window
    
    def loss_fn():
        """Highly optimized loss function"""
        if lambda_beta > 0:
            # Use independent beta_t variable
            current_beta_t = beta_t
        else:
            # Traditional method: beta_t = estR + eta
            current_beta_t = estR_tf + eta
        
        # Optimized matrix operations: using pre-calculated constants
        # 1. Linear predictor: X * beta_t (vectorized)
        linear_pred = tf.linalg.matvec(ppData_tf, current_beta_t)
        
        # 2. Adjusted cumulative baseline hazard: cumH + Xinn * xi (vectorized)
        xi_contribution = tf.linalg.matvec(XiData_tf, xi)
        adjusted_cumH = cQ_tf + xi_contribution
        
        # 3. Adjusted baseline hazard: hazards + xi
        adjusted_hazards = dq_tf + xi
        
        # Numerical stability: ensure adjusted hazards are positive
        adjusted_hazards = tf.maximum(adjusted_hazards, 1e-8)
        
        # Event-related calculations (vectorized)
        event_linear_pred = tf.reduce_sum(tf.gather(linear_pred, smallidx))
        log_hazards = tf.reduce_sum(tf.math.log(adjusted_hazards))
        
        # Risk set calculation (vectorized exponential operation)
        exp_linear_pred = tf.math.exp(linear_pred)
        risk_set_term = tf.reduce_sum(adjusted_cumH * exp_linear_pred)
        
        # Negative log-likelihood
        neg_log_likelihood = -(event_linear_pred + log_hazards - risk_set_term)
        
        # Vectorized L1 penalty calculation
        if lambda_beta > 0:
            # Fix: When using lambda_beta, eta = beta_t - estR, need to explicitly reflect this relationship in loss function
            # Previous implementation used independent eta variable for l1_eta, causing transfer failure
            eta_diff = beta_t - estR_tf
            l1_eta = lambda1 * tf.reduce_sum(tf.abs(eta_diff))
        else:
            l1_eta = lambda1 * tf.reduce_sum(tf.abs(eta))
            
        l1_xi = lambda2 * tf.reduce_sum(tf.abs(xi))
        l1_beta = lambda_beta * tf.reduce_sum(tf.abs(current_beta_t))
        
        # Total loss
        total_loss = neg_log_likelihood + l1_eta + l1_xi + l1_beta
        
        return total_loss
    
    # Optimizer
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    
    # Optimization loop
    if verbose:
        print("Optimization started...")
    
    # Ensure nsteps is integer
    nsteps = int(nsteps)
    
    for step in range(nsteps):
        with tf.GradientTape() as tape:
            loss_value = loss_fn()
        
        # Soft thresholding function definition
        def soft_threshold(x, threshold):
            """Optimized soft thresholding function using vectorized operations"""
            abs_x = tf.abs(x)
            return tf.sign(x) * tf.maximum(abs_x - threshold, 0.0)
        
        # Calculate gradients - include beta_t for true sparse optimization
        if lambda_beta > 0:
            # Fix: Only optimize beta_t and xi, eta is derived (determined by beta_t - estR)
            # Previous implementation optimized both eta and beta_t without constraint, which was wrong
            gradients = tape.gradient(loss_value, [beta_t, xi])
            optimizer.apply_gradients(zip(gradients, [beta_t, xi]))
            
            # Apply soft thresholding to all parameters to achieve sparsity
            threshold_beta = lambda_beta * 0.1  # Further reduce threshold to avoid over-sparsification
            beta_t.assign(soft_threshold(beta_t, threshold_beta))
            
            # eta is not an independent variable, no need to optimize or soft threshold, just update its value for subsequent statistics
            eta.assign(beta_t - estR_tf)
            
            threshold_xi = lambda2 * 0.05  # Further reduce threshold to avoid over-sparsification
            xi.assign(soft_threshold(xi, threshold_xi))
            
        else:
            gradients = tape.gradient(loss_value, [eta, xi])
            optimizer.apply_gradients(zip(gradients, [eta, xi]))
            
            # Apply soft thresholding even in non-sparse mode to get sparsity
            # Apply soft thresholding to eta
            threshold_eta = lambda1 * 0.01  # Milder thresholding
            eta_sign = tf.sign(eta)
            eta_abs = tf.abs(eta)
            eta_thresholded = eta_sign * tf.maximum(0.0, eta_abs - threshold_eta)
            eta.assign(eta_thresholded)
            
            # Apply soft thresholding to xi
            threshold_xi = lambda2 * 0.01  # Milder thresholding
            xi_sign = tf.sign(xi)
            xi_abs = tf.abs(xi)
            xi_thresholded = xi_sign * tf.maximum(0.0, xi_abs - threshold_xi)
            xi.assign(xi_thresholded)
        
        # Record loss
        loss_history.append(loss_value.numpy())
        
        # Print progress every 50 steps
        if verbose and (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{nsteps}, Loss: {loss_value.numpy():.6f}")
        
        # Optimized convergence check
        if step > convergence_window:
            # Dual check using loss change and parameter change
            recent_losses = loss_history[-convergence_window:]
            loss_change = max(recent_losses) - min(recent_losses)
            
            # Parameter change check
            if step > 0:
                # Calculate relative change of parameters
                eta_norm = tf.norm(eta)
                xi_norm = tf.norm(xi)
                
                if lambda_beta > 0:
                    beta_norm = tf.norm(beta_t)
                    total_norm = tf.sqrt(eta_norm**2 + xi_norm**2 + beta_norm**2)
                else:
                    total_norm = tf.sqrt(eta_norm**2 + xi_norm**2)
                
                # If loss change is small and parameter norm is stable, consider converged
                if loss_change < tolerance and total_norm > 0:
                    if verbose:
                        print(f"Converged at step {step + 1}, loss change: {loss_change:.6f}")
                    break
    
    # Calculate final results
    eta_final = eta.numpy()
    xi_final = xi.numpy()
    
    if lambda_beta > 0:
        # When lambda_beta > 0, beta_t has obtained sparsity directly through optimization
        beta_t_final = beta_t.numpy()
        # Recalculate eta to maintain consistency
        eta_final = beta_t_final - estR_np
    else:
        # When lambda_beta = 0, use traditional method
        beta_t_final = estR_np + eta_final
    
    # Calculate sparsity statistics
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
        print(f"Optimization Completed!")
        print(f"Final Loss: {convergence_info['final_loss']:.6f}")
        print(f"eta Non-zero Coeffs: {nonzero_eta}/{len(eta_final)} ({(1-convergence_info['sparsity_eta'])*100:.1f}%)")
        print(f"xi Non-zero Coeffs: {nonzero_xi}/{len(xi_final)} ({(1-convergence_info['sparsity_xi'])*100:.1f}%)")
        print(f"beta_t Non-zero Coeffs: {nonzero_beta}/{len(beta_t_final)} ({(1-convergence_info['sparsity_beta'])*100:.1f}%)")
    
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