# references: 
# [1] https://docs.ufpr.br/~danielsantos/ProbabilisticRobotics.pdf

import numpy as np
import sympy
from sympy.abc import alpha, x, y, v, w, R, theta
from sympy import symbols, Matrix
# from kalman_filters import ExtendedKalmanFilter_setup1

class ExtendedKalmanFilter_setup1:
    """Extended Kalman Filter
    for vehicle whose motion is modeled as eq. (5.9) in [1]
    and with observation of its 2d location (x, y)
    """
    x = None
    P = None
    R = None
    def __init__(self, x, P, H):
        """ 
        Args:
            x (numpy.array): state to estimate: [x_, y_, theta]^T
            P (numpy.array): estimation error covariance
        """
        px = symbols('p_{x}')
        py = symbols('p_{y}')
        pz = symbols('p_{z}')
        
        vx = symbols('v_{x}')
        vy = symbols('v_{y}')
        vz = symbols('v_{z}')
        
        q1 = symbols('q1')
        q2 = symbols('q2')
        q3 = symbols('q3')
        q4 = symbols('q4')
        
        ax = symbols('a_{x}')
        ay = symbols('a_{y}')
        az = symbols('a_{z}')
        wx = symbols('w_{x}')
        wy = symbols('w_{y}')
        wz = symbols('w_{z}')

        dt = symbols('dt')
        norm_w = symbols('\|w\|')
        aR1 = symbols('aR_{1}')
        aR2 = symbols('aR_{2}')
        aR3 = symbols('aR_{3}')
        q_prev = symbols('q_{k-1}')
        
        g = Matrix([
            [0],[0],[9.81]
        ])
        
        R = Matrix([[q1**2 + q2**2 + q3**2 + q4**2, 2*(q2*q3 - q1*q4), 2*(q1*q3 + q2*q4)],
            [2*(q2*q3 + q1*q4), q1**2 - q2**2 + q3**2 - q4**2, 2*(q3*q4 - q1*q2)],
            [2*(q2*q4 - q1*q3), 2*(q1*q2 + q3*q4), q1**2 - q2**2 - q3**2 + q4**2]
           ])
        Omega = Matrix([
            [0, wx, wy, wz],
            [-wz, 0, wx, wy],
            [wy, -wx, 0, wz],
            [-wx, -wy, -wz, 0]
        ])
        A = sympy.cos(norm_w*dt/2) * sympy.eye(4)
        B = (1/norm_w)*sympy.sin(norm_w*dt/2)
        qk = (A + B * Omega) * P
        self.fxu = Matrix([
            [px + vx*dt + 1/2 * (R[0,0]*ax-g[0] + R[0,1]*ay-g[1] + R[0,2]*az-g[2])*dt**2],
            [py + vy*dt + 1/2 * (R[1,0]*ax-g[0] + R[1,1]*ay-g[1] + R[1,2]*az-g[2])*dt**2],
            [pz + vz*dt + 1/2 * (R[2,0]*ax-g[0] + R[2,1]*ay-g[1] + R[2,2]*az-g[2])*dt**2],
            [vx + (R[0,0]*ax-g[0] + R[0,1]*ay-g[1] + R[0,2]*az-g[2])*dt],
            [vy + (R[1,0]*ax-g[0] + R[1,1]*ay-g[1] + R[1,2]*az-g[2])*dt],
            [vz + (R[2,0]*ax-g[0] + R[2,1]*ay-g[1] + R[2,2]*az-g[2])*dt],
            [qk[0]],
            [qk[1]],
            [qk[2]],
            [qk[3]],
        ])
        state_x = Matrix([px, py, pz, vx, vy, vz, q1, q2, q3, q4])
        control_input = Matrix([ax, ay, az, wx, wy, wz])

        self.F = self.fxu.jacobian(state_x)
        self.P = P
        self.H = H
        self.x = x

    def compute_norm_w(self, wx_, wy_, wz_):
        return np.sqrt(wx_**2 + wy_**2 + wz_**2)
        
    def predict(self, u, dt_, Q):
        """estimate x and P based on previous stete of x and control input u
        Args:
            u  (numpy.array): control input u
            dt (numpy.array): difference of current time and previous time
            Q  (numpy.array): process noise 
        """
        # propagate state x
        px_, py_, pz_ = self.x[:3]
        vx_, vy_, vz_ = self.x[3:6]
        q1_, q2_, q3_, q4_ = self.x[6:]
        ax_, ay_, az_, wx_, wy_, wz_ = u
        norm_w_ = self.compute_norm_w(wx_, wy_, wz_);
        fxu_values = {
            dt: dt_,
            px: px_,
            py: py_,
            pz: pz_,
            vx: vx_,
            vy: vy_,
            vz: vz_,
            q1: q1_, 
            q2: q2_, 
            q3: q3_, 
            q4: q4_,
            ax: ax_,
            ay: ay_,
            az: az_,
            wx: wx_,
            wy: wy_, 
            wz: wz_,
            norm_w: norm_w_
        }
        self.fxu.evalf(subs=fxu_values)
        # predict state vector x
        # self.x = np.array(self.fxu.evalf(subs=fxu_values)).astype(float)
        
        # # predict state covariance matrix P
        # F_jacobian = np.array(self.F.evalf(subs=f_values)).astype(float)
        # self.P = F_jacobian @ self.P @ F_jacobian.T + Q
        
    def update(self, z, R):
        """update x and P based on observation of (x_, y_)
        Args:
            z (numpy.array): measurement for [x_, y_]^T
            R (numpy.array): measurement noise covariance
        """
        # compute Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + R)

        # update state x
        z_ = np.dot(self.H, self.x)  # expected observation from the estimated state
        self.x = self.x + K @ (z - z_)

        # update covariance P
        self.P = self.P - K @ self.H @ self.P


class ExtendedKalmanFilter_setup3:
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
