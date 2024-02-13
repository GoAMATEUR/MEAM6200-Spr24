import numpy as np
from scipy.spatial.transform import Rotation

class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2

        # STUDENT CODE HERE
        self.k_p = 8.5 * np.eye(3)
        self.k_d = 5 * np.eye(3)
        self.k_r = 125 * np.eye(3)
        self.k_w = 13 * np.eye(3)
        
        self.gamma = self.k_drag / self.k_thrust

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))
        # STUDENT CODE HERE
        x_ddot_des = flat_output['x_ddot'] -\
            self.k_d @ (state['v'] - flat_output['x_dot']) -\
                self.k_p @ (state['x'] - flat_output['x'])
        F_des = self.mass * x_ddot_des.reshape(3, 1) +\
            np.array([0, 0, self.mass * self.g]).reshape(3, 1)

         # 2. Find u_1
        R = Rotation.from_quat(state['q']).as_matrix()
        b3 = R @ np.array([0, 0, 1]).reshape(-1, 1) # (3, 1)
        # u = np.zeros([4, 1])
        u_1 = b3.T @ F_des # (1, 1)
        # 3. Find R_des
        b3_des = F_des / np.linalg.norm(F_des)
        phi = flat_output['yaw']
        a_phi = np.array([np.cos(phi), np.sin(phi), 0]).reshape(-1, 1)
        b2_des = np.cross(b3_des, a_phi, axis=0)
        b2_des = b2_des / np.linalg.norm(b2_des)
        R_des = np.hstack([np.cross(b2_des, b3_des,axis=0), b2_des, b3_des])
        # 4. Error e_R
        e_R = (R_des.T @ R - R.T @ R_des) / 2
        e_R = np.array([e_R[2,1],e_R[0,2],e_R[1,0]])
        e_w = state['w']
        u_2 = self.inertia @ (-self.k_r @ e_R - self.k_w @ e_w).reshape(-1, 1)
        
        motor_forces = np.linalg.inv(np.array([
            [1, 1, 1, 1],
            [0, self.arm_length, 0, -self.arm_length],
            [-self.arm_length, 0, self.arm_length, 0],
            [self.gamma, -self.gamma, self.gamma, -self.gamma]
        ])) @ np.concatenate((u_1, u_2), axis=0)
        motor_forces[motor_forces < 0] = 0
        cmd_moment = u_2
        motor_forces[motor_forces< 0] = 0
        cmd_motor_speeds = np.sqrt(motor_forces / self.k_thrust).reshape(4)
        cmd_motor_speeds = np.clip(cmd_motor_speeds, self.rotor_speed_min, self.rotor_speed_max)
        # actual_forces = np.power(cmd_motor_speeds, 2) * self.k_thrust
        cmd_thrust = np.sum(motor_forces)
        cmd_q = Rotation.from_matrix(R_des).as_quat()
        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input
