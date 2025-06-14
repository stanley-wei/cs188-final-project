import numpy as np
import torch
from behavior_cloning import BCModel


class BCPolicy:
    def __init__(self, model_path="best_bc_model.pt"):
        """
        Initialize the Behavior Cloning policy with a trained model.

        Args:
            model_path: Path to the saved model weights
        """
        # Load the model architecture
        self.model = BCModel(input_dim=15, action_dim=7)

        # Load the trained weights
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def get_action(self, obs):
        """
        Get action from the policy based on current observation.

        Args:
            obs: Dictionary containing observation fields

        Returns:
            numpy array: Action vector (7-dimensional)
        """
        eef_pos = obs["robot0_eef_pos"]
        eef_quat = obs["robot0_eef_quat"]
        gripper_pos = obs["robot0_gripper_qpos"][0:1]
        obj_pos = obs["SquareNut_pos"]
        obj_quat = obs["SquareNut_quat"]

        state = np.concatenate([eef_pos, eef_quat, gripper_pos, obj_pos, obj_quat])

        # Convert to tensor and get prediction
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
            action_tensor = self.model(state_tensor)

        # Convert back to numpy array and return
        # print("Action:", action_tensor.squeeze(0).numpy())
        return action_tensor.squeeze(0).numpy()
