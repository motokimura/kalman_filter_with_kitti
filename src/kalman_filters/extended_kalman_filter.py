# references: 
# [1] https://docs.ufpr.br/~danielsantos/ProbabilisticRobotics.pdf

import numpy as np


class ExtendedKalmanFilter:
    """Extended Kalman Filter
    for vehicle whose motion is modeled as eq. (5.9) in [1]
    and with observation of its 2d location (x, y)
    """
    def __init__(self, x, P):
        """ 
        Args:
            x (numpy.array): state to estimate: [x_, y_, theta]^T
            P (numpy.array): estimation error covariance
        """
        self.x = x  #  [3,]
        self.P = P  #  [3, 3]

    def update(self, z, Q):
        """update x and P based on observation of (x_, y_)
        Args:
            z (numpy.array): obsrervation for [x_, y_]^T
            Q (numpy.array): observation noise covariance
        """
        # compute Kalman gain
        H = np.array([
            [1., 0., 0.],
            [0., 1., 0.]
        ])  # Jacobian of observation function

        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + Q)

        # update state x
        x, y, theta = self.x
        z_ = np.array([x, y])  # expected observation from the estimated state
        self.x = self.x + K @ (z - z_)

        # update covariance P
        self.P = self.P - K @ H @ self.P

    def propagate(self, u, dt, R):
        """propagate x and P based on state transition model defined as eq. (5.9) in [1]
        Args:
            u (numpy.array): control input: [v, omega]^T
            dt (float): time interval in second
            R (numpy.array): state transition noise covariance
        """
        # propagate state x
        x, y, theta = self.x
        v, omega = u
        r = v / omega  # turning radius

        dtheta = omega * dt
        dx = - r * np.sin(theta) + r * np.sin(theta + dtheta)
        dy = + r * np.cos(theta) - r * np.cos(theta + dtheta)

        self.x += np.array([dx, dy, dtheta])

        # propagate covariance P
        G = np.array([
            [1., 0., - r * np.cos(theta) + r * np.cos(theta + dtheta)],
            [0., 1., - r * np.sin(theta) + r * np.sin(theta + dtheta)],
            [0., 0., 1.]
        ])  # Jacobian of state transition function

        self.P = G @ self.P @ G.T + R
