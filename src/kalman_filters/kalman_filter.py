
import numpy as np
from numpy import dot, zeros, eye
from numpy.linalg import inv

class KalmanFilter:
    '''
    Kalman filtering, also known as linear quadratic estimation (LQE), is an algorithm that uses a series of measurements
    observed over time, containing statistical noise and other inaccuracies,
    and produces estimates of unknown variables that tend to be more accurate than those based on a single measurement
    alone, by estimating a joint probability distribution over the variables for each time frame.
    '''
    def __init__(self, dim_x, dim_z):
        #self.debug_flag = config.DEBUG_FLAG
        self.debug_flag = False
        if self.debug_flag:
            print("KalmanFilter::Init")
            print("dim_x: {}, dim_z: {}".format(dim_x, dim_z))

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = zeros((dim_x, 1))
        self.P = eye(dim_x)
        self.Q = eye(dim_x)
        self.F = eye(dim_x)
        self.H = zeros((dim_z, dim_x))
        self.R = eye(dim_z)
        self.M = zeros((dim_z, dim_z))

        self._I = eye(dim_x)  # This helps the I matrix to always be compatible to the state vector's dim
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)
        if self.debug_flag:
            print("KalmanFilter::Init, x_prior: {}, P_prior: {}".format(self.x_prior, self.P_prior))

    def predict(self):
        '''
        Predict next state (prior) using the Kalman filter state propagation
        equations.
        '''
        if self.debug_flag:
            print("KalmanFilter::predict")
        self.x = dot(self.F, self.x)                                    # x = Fx
        self.P = dot(self.F, dot(self.P, self.F.T)) + self.Q            # P = FPF' + Q
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

        if self.debug_flag:
            print("KalmanFilter::predict, x_prior: {}, P_prior: {}".format(self.x_prior, self.P_prior))

    def update(self, z):
        '''
        At the time step k, this update step computes the posterior mean x and covariance P
        of the system state given a new measurement z.
        '''
        if self.debug_flag:
            print("KalmanFilter::update, z: {}".format(z))
        # y = z - Hx (Residual between measurement and prediction)
        y = z - np.dot(self.H, self.x)
        PHT = dot(self.P, self.H.T)

        # S = HPH' + R (Project system uncertainty into measurement space)
        S = dot(self.H, PHT) + self.R

        # K = PH'S^-1  (map system uncertainty into Kalman gain)
        K = dot(PHT, inv(S))
        if self.debug_flag:
            print("KalmanFilter::update, K.shape: {}, K: {}".format(K.shape, K))

        # x = x + Ky  (predict new x with residual scaled by the Kalman gain)
        self.x = self.x + dot(K, y)

        if self.debug_flag:
            print("KalmanFilter::update, x.shape: {}, x :{}".format(self.x.shape, self.x))

        # P = (I-KH)P
        I_KH = self._I - dot(K, self.H)
        self.P = dot(I_KH, self.P)

        if self.debug_flag:
            print("KalmanFilter::update, P.shape: {}, P :{}".format(self.P.shape, self.P))