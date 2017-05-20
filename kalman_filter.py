# -*- coding: utf-8 -*-
"""
Created on Mon May  1 15:48:18 2017

@author: Ratnadeepb
"""

import numpy as np
import numpy.linalg as la

# Convolution of Gaussians
# Measurement Update

def update(mean1, var1, mean2, var2):
    new_var = 1 / ((1 / var1) + (1 / var2))
    new_mean = new_var * ((mean1 / var1) + (mean2 / var2))
    return [new_mean, new_var]

# Sum of Gaussians
# Motion update

def predict(mean_pos, var_pos, mean_move, var_move):
    m = mean_move + mean_pos
    v = var_move + var_pos
    return [m, v]

### Kalman Filtering

def motion_update(F, B, P, Q, x_t, u):
    x_t = F @ x_t + B @ u
    sigma_t = F @ P @ F.T + Q
    
    return [x_t, sigma_t]

def measurement_update(x_now, P_now, H, Z, R):
    K = P_now @ H.T @ la.inv(H @ P_now @ H.T + R)
    x_t = x_now + K @ (Z.T - H @ x_now)
    sigma_t = P_now - K @ H @ P_now
    
    return [x_t, sigma_t]


if __name__ == "__main__":
    # Change in time
    dt = 0.1
    # Initial Covariance
    P = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1000, 0], [0, 0, 0, 1000]])
    # Additional Motion at interval dt
    u = np.array([0, 0, 0, 0])
    # Motion transition
    B = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # Variance of motion
    Q = np.array([[0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0.1]])
    # State Transition
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    # Mapping Z (2D measurement of x, y) to X (4D - x, y, x_dot, y_dot)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    # Variance of measurement
    R = np.array([[0.1, 0], [0, 0.1]])

    # Series of measurements    
    measurements = np.array([[1., 4.], [6., 0.], [11., -4.], [16., -8.]])
    # Initial Position
    x_init = np.array([-4, 8])
    
    # Applying the filter
    x_t = np.array([x_init[0], x_init[1], 0, 0])
    
    for i in range(measurements.shape[0]):
        Z = measurements[i]
        x_t, P = motion_update(F, B, P, Q, x_t, u)
        x_t, P = measurement_update(x_t, P, H, Z, R)
        print("State - x_t: ", x_t)