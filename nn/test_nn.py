import datetime
import numpy as np
import robosuite as suite
from bc_policy import BCPolicy
from robosuite.utils.placement_samplers import UniformRandomSampler

placement_initializer = UniformRandomSampler(
    name="FixedOriSampler",
    mujoco_objects=None,            
    x_range=[-0.115, -0.11],       
    y_range=[0.05, 0.225],
    rotation=np.pi,
    rotation_axis="z",
    ensure_object_boundary_in_range=False,
    ensure_valid_placement=False,
    reference_pos=(0,0,0.82),
    z_offset=0.02,
)

# create environment instance
env = suite.make(
    env_name="NutAssemblySquare", 
    robots="Panda", 
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    ignore_done=True,
    # placement_initializer=placement_initializer,
    camera_names=["agentview"],
)

success_rate = 0
trials = 50
file = open('times.csv', 'a')
# reset the environment
for _ in range(trials):
    obs = env.reset()
    policy = BCPolicy('best_bc_model.pt')
    
    time = datetime.datetime.now()
    success = False
    for i in range(1000):
        action = policy.get_action(obs)
        obs, reward, done, info = env.step(action)  # take action in the environment
        env.render()  # render on display
        
        if reward == 1.0:
            success_rate += 1
            success = True
            completion_time = (datetime.datetime.now() - time).total_seconds()
            print('completion time:', completion_time)
            break
        
    if success:
        file.write(str(completion_time) + '\n')
    
success_rate /= trials
print('success rate:', success_rate)
file.close()

