import numpy as np

class CanonicalSystem:
    """
    Skeleton of the discrete canonical dynamical system.
    """
    def __init__(self, dt: float, ax: float = 1.0):
        """
        Args:
            dt (float): Timestep duration.
            ax (float): Gain on the canonical decay.
        """
        # Initialize time parameters
        self.dt: float = dt
        self.ax: float = ax
        self.run_time: float = 1.0  # 1.0: set total runtime
        self.timesteps: int = int(self.run_time / dt)
        self.x: float = None  # phase variable
        
        self.reset()


    def reset(self) -> None:
        """
        Reset the phase variable to its initial value.
        """
        self.x = 1.0


    def step(self, tau: float = 1.0, error_coupling: float = 1.0) -> float:
        """
        Advance the phase by one timestep.

        Returns:
            float: Updated phase value.
        """
        self.x += (-self.ax * self.x * error_coupling) * self.dt * tau
        return self.x


    def rollout(self, tau: float = 1.0, ec: float = 1.0) -> np.ndarray:
        """
        Generate the entire phase sequence.

        Returns:
            np.ndarray: Array of phase values over time.
        """
        x_arr = np.zeros(self.timesteps)
        self.reset()
        for i in range(self.timesteps):
            x_arr[i] = self.step(tau, ec)
        return x_arr


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    cs = CanonicalSystem(0.01, 5)
    rollout = cs.rollout()
    print(rollout)
    
    plt.scatter(np.linspace(0, 1, rollout.shape[0]), rollout)
    plt.show()
