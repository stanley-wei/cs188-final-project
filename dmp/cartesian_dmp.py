import numpy as np
from scipy import interpolate, linalg
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from canonical_system import CanonicalSystem
import utils


class CartesianDMP:

    def __init__(
        self,
        n_bfs: int,
        dt: float = 0.01,
        az: float = 25.0,
        bz: float = None
    ):
        """
        Args:
            n_bfs (int): Number of basis functions per dimension.
            dt (float): Timestep duration.
            az (float|array): Attractor gain.
            bz (float|array): Damping gain.
        """
        # TODO: initialize parameters
        self.n_bfs: int = n_bfs
        self.dt: float = dt
        self.run_time = 1.0
        
        self.az: np.ndarray = az
        self.bz: np.ndarray = bz
        
        self.wp = None  # Position DMP weights
        self.wq = None  # Orientation DMP weights
        
        self.Dp = np.zeros((3,3))   # Position amplitude scaling matrix
        self.Do = np.zeros((3,3))   # Orientation amplitude scaling matrix
        
        self.cs: CanonicalSystem = CanonicalSystem(self.dt, 3)
        self.reset_state()
        

    def reset_state(self) -> None:
        """
        Reset canonical system state.
        """
        self.cs.reset()

    def _psi(self, s):
        """
        Compute basis functions.
        """
        centers = np.linspace(0, self.run_time, self.n_bfs)
        widths = 3000 * ((self.n_bfs ** 1.5) / 50)
        psi = np.exp(-widths * np.power(s - centers, 2)) 
           
        return psi / np.sum(psi)

    def _force_p(self, s, Dp = None):
        """
        Compute forcing function from position weights.
        """
        D = self.Dp if Dp is None else Dp
        return D @ np.dot(self._psi(s), self.wp)

    def _force_q(self, s, Do = None):
        """
        Compute forcing function from orientation weights.
        """
        D = self.Do if Do is None else Do
        return D @ np.dot(self._psi(s), self.wq)

    def imitate(self, x_des, q_des: np.ndarray, tau=1.0) -> np.ndarray:
        """
        Learn DMP weights from a demonstration.

        Args:
            x_des (np.ndarray): Desired position trajectory, shape (T, 3).
            q_des (np.ndarray): Desired rotation trajectory (as quaternion), shape (T, 4).
            tau (float): Time constant, default = 1.0.

        Returns:
            np.ndarray: Interpolated demonstration positions (T' x 3).
            np.ndarray: Interpolated demonstration orientations (T' x 3 x 3).
        """
        x_des = self.find_position_weights(x_des, tau)
        R_des = self.find_orientation_weights(q_des, tau)
        
        return x_des, R_des

    def find_position_weights(self, x_traj, tau):
        """
        Learn DMP weights for generating position trajectories.

        Args:
            x_traj (np.ndarray): Desired position trajectory, shape (T, 3).
            tau (float): Time constant.

        Returns:
            np.ndarray: Interpolated demonstration orientations (T' x 3 x 3).
        """
        stamps = np.linspace(0, self.run_time, x_traj.shape[0])
        timestamps = np.arange(0, self.run_time, self.dt)

        x_des = np.swapaxes(np.asarray([np.interp(timestamps, stamps, x_traj[:, i]) for i in range(3)]), 0, 1)
        dx_des = np.gradient(x_des, axis=0) / self.dt
        ddx_des = np.gradient(dx_des, axis=0) / self.dt
        
        x_start = x_traj[0]
        x_goal = x_traj[-1]
        self.Dp = np.diag(x_goal - x_start)

        self.reset_state()
        phases = self.cs.rollout()
        psi = np.asarray([self._psi(phase) for phase in phases])
        
        out = pow(tau, 2) * ddx_des + self.az * tau * dx_des - self.az * self.bz * (x_goal - x_des)
        f_target = np.asarray([np.linalg.inv(self.Dp) @ fp for fp in out])

        self.wp = np.linalg.lstsq(psi, f_target, rcond=None)[0]
        return x_des
        

    def find_orientation_weights(self, q_traj, tau):
        """
        Learn DMP weights for generating orientation trajectories.

        Args:
            q_traj (np.ndarray): Desired rotation trajectory (as quaternion), shape (T, 4).
            tau (float): Time constant.

        Returns:
            np.ndarray: Interpolated demonstration orientations (T' x 3 x 3).
        """
        stamps = np.linspace(0, self.run_time, q_traj.shape[0])
        slerp = Slerp(stamps, R.from_quat(q_traj))
        
        timestamps = np.arange(0, self.run_time, self.dt)
        R_des = slerp(timestamps).as_matrix()
        
        R_0 = R.from_quat(q_traj[0]).as_matrix()
        R_goal = R.from_quat(q_traj[-1]).as_matrix()
        self.Do = np.diag(utils.log_so3(R_goal @ R_0.transpose()))        
        
        dR_des = np.gradient(R_des, axis=0) / self.dt
        w_cross_des = np.asarray([dR_des[i] @ R_des[i].transpose() for i in range(timestamps.shape[0])])
        w_des = np.asarray([utils.from_cross_matrix(mat) for mat in w_cross_des])
        dw_des = np.gradient(w_des, axis=0) / self.dt
    
        self.reset_state()
        phases = self.cs.rollout()
        psi = np.asarray([self._psi(phase) for phase in phases])
    
        out = dw_des * pow(tau, 2) + self.az * tau * w_des - self.az * self.bz * np.asarray([utils.log_so3(R_goal @ R_mat.transpose()) for R_mat in R_des])
        f_target = np.asarray([np.linalg.inv(self.Do) @ fo for fo in out])
        
        self.wq = np.linalg.lstsq(psi, f_target, rcond=None)[0]
        return R_des


    def rollout(
        self,
        tau: float = 1.0,
        error: float = 0.0,
        x0: np.ndarray = None,
        q0: np.ndarray = None,
        goal_x: np.ndarray = None,
        goal_q: np.ndarray = None
    ) -> np.ndarray:
        """
        Generate a new trajectory from the DMP.

        Args:
            tau (float): Temporal scaling.
            error (float): Feedback coupling.
            x0 (np.ndarray): Initial position, shape (3).
            q0 (np.ndarray): Initial orientation (as quaternion), shape (4).
            goal_x (np.ndarray): Goal position, shape (3).
            goal_q (np.ndarray): Goal orientation (as quaternion), shape (4).
            
        Returns:
            np.ndarray: Generated position trajectory (T x 3).
            np.ndarray: Generated orientation trajectory, as quaternion (T x 4).
        """

        self.reset_state()
        phases = self.cs.rollout(tau)
        timesteps = np.arange(0, self.run_time, self.dt)

        x_rollout = np.zeros((timesteps.shape[0], 3))
        R_rollout = np.zeros((timesteps.shape[0], 3, 3))
        
        x = x0
        dx = np.zeros(3)
        ddx = np.zeros(3)
        
        R_mat = R.from_quat(q0).as_matrix()
        eta = np.zeros(3)
        eta = np.zeros(3)
           
        x_rollout[0] = x
        R_rollout[0] = R_mat
        
        goal_R = R.from_quat(goal_q).as_matrix()
        Dp = np.diag(goal_x - x)
        Do = np.diag(utils.log_so3(goal_R @ R_mat.transpose()))    
           
        for i in range(1, timesteps.shape[0]):
            ddx = self.az * (self.bz * (goal_x - x) - dx) + self._force_p(phases[i-1], Dp)
            dx += self.dt * ddx / tau
            x += self.dt * dx / tau

            deta = self.az * (self.bz * utils.log_so3(goal_R @ R_mat.transpose()) - eta) + self._force_q(phases[i-1], Do)
            eta += self.dt * deta / tau
            R_mat = linalg.expm(self.dt * utils.to_cross_matrix(eta) / tau) @ R_mat
            
            x_rollout[i] = x
            R_rollout[i] = R_mat
        
        q_rollout = R.from_matrix(R_rollout).as_quat()
        return x_rollout, q_rollout
        
