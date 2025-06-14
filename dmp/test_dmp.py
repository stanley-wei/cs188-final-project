import datetime
import numpy as np
import robosuite as suite
from dmp_policy import DMPPolicyWithPID
from robosuite.utils.placement_samplers import UniformRandomSampler


# create environment instance
env = suite.make(
    env_name="NutAssemblySquare", 
    robots="Panda", 
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    ignore_done=True,
    camera_names=["agentview"],
)

num_trials = 50
success_rate = 0
file = open('times_old.csv', 'a')
# reset the environment
for _ in range(num_trials):
    obs = env.reset()
    policy = DMPPolicyWithPID(obs['SquareNut_pos'], obs['SquareNut_quat'], obs['robot0_eef_pos'], obs['robot0_eef_quat'], debug=False)

    time = datetime.datetime.now()
    success = False
    for _ in range(2500):
        action = policy.get_action(obs['robot0_eef_pos'], obs['robot0_eef_quat'])
        if action[-1] == 1: # stop early if the robot has completed all trajectories
            if (datetime.datetime.now() - time).total_seconds() > 2:
                break
        else:
            time = datetime.datetime.now()
        obs, reward, done, info = env.step(action[:-1])  # take action in the environment
        env.render()  # render on display
        if reward == 1.0:
            success_rate += 1
            success = True
            print('success')
            break
    
    if success:
        file.write(str((datetime.datetime.now() - time).total_seconds()) + "\n")

success_rate /= num_trials
print('success rate:', success_rate)
