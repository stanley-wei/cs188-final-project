import datetime
import numpy as np
from collections import defaultdict
from scipy.spatial.transform import Rotation as R

from cartesian_dmp import CartesianDMP
from pid import PID, RotationPID


class DMPPolicyWithPID:
    """
    A policy that follows a demonstrated path with DMPs and PID control.

    The demonstration is split into segments based on grasp toggles.  
    The first segment's endpoint is re-targeted to a new object pose.
    Subsequent segments replay the original DMP rollouts.

    Args:
        square_obs (dict): 'SquareNut_pos' observed
        demo_path (str): path to .npz file with demo data.
        dt (float): control timestep.
        n_bfs (int): number of basis functions per DMP.
    """
    def __init__(self, square_pos, square_quat, robot_pos, robot_quat, demo_path='final_project_default_data/demos.npz', dt=0.01, n_bfs=30, debug=False):
        self.dt = dt
        demo = self.find_nearest_demo(square_quat, demo_path)

        # Extract trajectories and grasp
        ee_pos = demo['obs_robot0_eef_pos']  # (T,3)
        ee_rot = R.from_quat(demo['obs_robot0_eef_quat'])
        ee_grasp = demo['actions'][:, -1:].astype(int)  # (T,1)
        segments = self.detect_grasp_segments(ee_grasp)

        # Compute offset for first segment to new object pose
        demo_obj_pos = demo['obs_object'][0, :3]
        demo_obj_rot = R.from_euler('xyz', demo['obs_object'][0, 3:6])
        start, end = segments[0]
        
        offset = ee_pos[end-1] - demo_obj_pos
        offrot = ee_rot[end-1] * demo_obj_rot.inv()
        
        # Find new target pose
        new_obj_pos = square_pos                
        new_obj_rot = R.from_quat(square_quat)
        
        new_goal_pos = new_obj_pos + offset
        new_goal_rot = ee_rot[end-1].as_quat()
        
        self.trajectories = []
        for i in range(len(segments)):
            start, end = segments[i]
            
            traj_position = ee_pos[segments[i][0]:segments[i][1]-1]
            traj_orientation = np.asarray([q.as_quat() for q in ee_rot[start:end-1]])
            
            dmp = CartesianDMP(n_bfs=n_bfs, dt=dt, az=25.0, bz=1.0)
            dmp.imitate(traj_position, traj_orientation)
            
            if i==0:
                x_rollout, q_rollout = dmp.rollout(x0=robot_pos, q0=robot_quat, goal_x=new_goal_pos, goal_q=new_goal_rot)
            elif i== 1:
                x_rollout, q_rollout = dmp.rollout(x0=new_goal_pos, q0=new_goal_rot, goal_x=ee_pos[end-1], goal_q=ee_rot[end-1].as_quat())
            else:
                x_rollout, q_rollout = dmp.rollout(x0=ee_pos[start], q0=ee_rot[start].as_quat(), goal_x=ee_pos[end-1], goal_q=ee_rot[end-1].as_quat())
            
            trajectory_p = x_rollout
            trajectory_q = q_rollout
            self.trajectories.append(np.concatenate((trajectory_p, trajectory_q), axis=1))

        self.kp = 10
        self.ki = 15
        self.kd = 0.1
        
        self.r_kp = 1.0
        self.r_ki = 0.4
        self.r_kd = 0.1

        self.p_controller = PID(self.kp, self.ki, self.kd, self.trajectories[0][0][:3])
        self.q_controller = RotationPID(self.r_kp, self.r_ki, self.r_kd, self.trajectories[0][0][3:])
                
        self.current_segment = 0
        self.traj_index = 0
        self.grasp = ee_grasp[0]
        
        self.thresh_p = [0.003, 0.01, 0.01]
        self.thresh_q = [0.05, 0.05, 0.05]
        self.stuck_counter = 0
        self.transition_timer = datetime.datetime.now()
        
        self.debug = debug
        

    def find_nearest_demo(self, square_quat, demo_path):
        try:
            flat_data = np.load(demo_path)
            demos = defaultdict(lambda: {})  # demo_id -> { field_path: data }
            for key in flat_data.files:
                parts = key.split('_', 2)  # Expect format: demo_0_fieldname
                if len(parts) < 3:
                    print(f"Skipping malformed key: {key}")
                    continue
                demo_id = f"{parts[0]}_{parts[1]}"  # e.g., demo_0
                field_name = parts[2]  # reconstruct path
                demos[demo_id][field_name] = flat_data[key]
        except Exception as e:
            print(f"Error reconstructing data: {e}")
            exit(1)

        square_q = R.from_quat(square_quat)
        errors = np.asarray([(R.from_quat(demos[key]['obs_object'][0][3:7]) * square_q.inv()).magnitude() for key in demos.keys()])
        min_idx = np.argmin(errors)
        return demos[f'demo_{min_idx}']

    
    def quat_to_rpy(self, quat):
        euler = R.from_quat(quat).as_euler('xyz', degrees=False)
        for i in range(3):
            while np.abs(euler[i]) > (2 * np.pi):
                euler[i] -= np.sign(euler) * (2 * np.pi)
            if np.abs(euler[i]) > np.pi:
                if euler[i] < 0:
                    euler[i] = (2 * np.pi) + euler[i]
                else:
                    euler[i] = -(2 * np.pi) + euler[i]
        return euler

    def rot_distance(self, q1, q2):
        quat1 = R.from_quat(q1)
        quat2 = R.from_quat(q2)
        
        return (quat1 * quat2.inv()).magnitude()
        

    def detect_grasp_segments(self, grasp_flags: np.ndarray) -> list:
        """
        Identify segments based on grasp toggles.

        Args:
            grasp_flags (np.ndarray): (T,1) array of grasp signals.

        Returns:
            List[Tuple[int,int]]: start and end indices per segment.
        """
        # TODO: implement boundary detection
        diff = np.diff(np.concatenate((np.asarray([[0]]), grasp_flags, np.asarray([[0]]))), axis=0)
        change_indices = np.sort(np.concatenate((np.argwhere(diff > 0), np.argwhere(diff < 0)), axis=0)[:, 0])
        segments = [(change_indices[i], change_indices[i+1]-1) for i in range(0, len(change_indices)-1)]
        self.start_grasp = grasp_flags[0]
        return segments


    def get_action(self, robot_eef_pos: np.ndarray, robot_eef_quat: np.ndarray) -> np.ndarray:
        """
        Compute next action for the robot's end-effector.

        Args:
            robot_eef_pos  (np.ndarray): Current end-effector position [x,y,z].
            robot_eef_quat (np.ndarray): Current end-effector orientation [x,y,z].

        Returns:
            np.ndarray: Action vector [dx,dy,dz,droll,dpitch,dyaw,grasp].
        """
        
        # Pause between trajectories to ensure sucessful grab
        if self.traj_index == 0 and (datetime.datetime.now() - self.transition_timer).total_seconds() < 0.2:
            action = np.zeros((8))
            action[6] = self.grasp
            return action
        
        # Finished all trajectories
        if self.current_segment == len(self.trajectories):
            action = np.zeros((8))
            action[6] = self.grasp
            action[7] = 1
            return action    
  
        while True:    
            goal = self.trajectories[self.current_segment][self.traj_index]
            
            p_target_reached = np.linalg.norm(goal[:3] - robot_eef_pos) < self.thresh_p[self.current_segment]
            q_target_reached = self.rot_distance(goal[3:], robot_eef_quat) < self.thresh_q[self.current_segment]
            
            if self.debug:
                print(f'{self.current_segment} {self.traj_index}:\t{p_target_reached}\t{q_target_reached}\t{np.concatenate((robot_eef_pos,self.quat_to_rpy(robot_eef_quat)))}\t{goal}')
            
            if (p_target_reached and q_target_reached) or self.stuck_counter > 40:
                self.traj_index += 1
                self.stuck_counter = 0
                
                # End of current trajectory
                if self.traj_index == len(self.trajectories[self.current_segment]):
                    self.current_segment += 1
                    self.traj_index = 0
                    self.grasp = 1 if self.grasp == -1 else -1
                    
                    if self.current_segment == len(self.trajectories):
                        return np.zeros((8))
                    
                    self.p_controller.reset(self.trajectories[self.current_segment][self.traj_index][:3])
                    self.q_controller.reset(self.trajectories[self.current_segment][self.traj_index][3:])
                    self.transition_timer = datetime.datetime.now()
                    
                    action = np.zeros((8))
                    action[6] = self.grasp
                    return action
                
                self.p_controller.reset(self.trajectories[self.current_segment][self.traj_index][:3])
                self.q_controller.reset(self.trajectories[self.current_segment][self.traj_index][3:])
                continue
            else:
                self.stuck_counter += 1

            update = self.p_controller.update(robot_eef_pos, self.dt)
            rot_update = self.q_controller.update(robot_eef_quat, self.dt)
            break
        
        action = np.zeros((8))
        action[0:3] = update
        action[3:6] = rot_update
        action[6] = self.grasp
        return action
        
