import numpy as np

class WaypointTraj(object):
    """

    """
    def __init__(self, points: np.ndarray):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission. For a waypoint
        trajectory, the input argument is an array of 3D destination
        coordinates. You are free to choose the times of arrival and the path
        taken between the points in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Inputs:
            points, (N, 3) array of N waypoint coordinates in 3D
        """
        self._points = points
        self._N = self._points.shape[0]
        self._speed = 1.0
        self._seg_times = None
        self._seg_speeds = None
        self._poly_coeffs = None
        self._generate_traj()
        
    
    def _generate_traj(self):
        seg_vectors = self._points[1:, :] - self._points[:-1, :] # (N-1, 3)
        seg_distances = np.linalg.norm(seg_vectors, axis=1)  # (N-1,)
        seg_directions = seg_vectors / seg_distances[:, np.newaxis] # (N-1, 3)
        self._seg_speeds = self._speed * seg_directions # (N-1, 3)
        self._seg_times = seg_distances / self._speed # (N-1)
        # Find traj coeffs
        self._poly_coeffs = np.zeros((self._N - 1, 3, 2))
        self._poly_coeffs[:, :, 0] = seg_vectors / self._seg_times[:, np.newaxis]
        self._poly_coeffs[:, :, 1] = self._points[:-1, :]
        # return

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0
        
        # determin time seg
        # if t > np.sum()
        time_seg = 0
        valid_time = False
        for i in range(self._N-1):
            time_seg += self._seg_times[i]
            if time_seg > t:
                time_seg -= self._seg_times[i]
                valid_time = True
                break
        if valid_time:
            x = self._poly_coeffs[i] @ np.array([t-time_seg, 1])
            x_dot = self._seg_speeds[i]
        else:
            x = self._points[-1]
            x_dot = np.zeros(3)
        # print(x)
        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output

if __name__ == "__main__":
    waypoints = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [1, 1, 1]])
    traj = WaypointTraj(waypoints)
    update = traj.update(5)
    print(update)