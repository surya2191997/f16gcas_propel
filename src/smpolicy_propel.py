from spinup.utils.run_utils import ExperimentGrid
from spinup import ddpg_pytorch
from spinup.algos.pytorch.ddpg.core import *
from spinup.algos.pytorch.ddpg.ddpg import *
import gym
import argparse

import subprocess

import csaf
import csaf.config as cconf
import csaf.system as csys


# function for setting up environment
def env_fn():
    my_conf = cconf.SystemConfig.from_toml(
        "/home/sdwivedi/csaf_architecture/examples/f16/f16_simple_config.toml")  
    def ground_collision_condition(cname, outs):
        """ground collision premature termnation condition"""
        return cname == "plant" and outs["states"][11] <= 0.0
    my_system = csys.System.from_config(my_conf)
    my_env = csys.SystemEnv("autopilot", my_system, terminating_conditions=ground_collision_condition)
    return my_env 


def train_ddpg(args):
    eg = ExperimentGrid("train_ddpg")
    eg.add('env_fn', env_fn)
    eg.add('seed', [0])
    eg.add('epochs', 50)    
    eg.add('gamma', 0.97)     
    eg.add('steps_per_epoch', 1000)
    eg.add('save_freq', 1)
    eg.add('max_ep_len', 2100)   
    eg.add('update_after', 1000)
    eg.add('ac_kwargs:hidden_sizes', [(256, 256)], 'hid')
    eg.run(ddpg_pytorch)

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", type=int, default=1)
parser.add_argument('--num_runs', type=int, default=0)
parser.add_argument('--env_name', type=str, default="Safexp-PointGoal1-v0")
parser.add_argument('--exp_name', type=str, default='ddpg-9gamma-rawruntil')
args = parser.parse_args()
train_ddpg(args)


import sys
sys.path.insert(0, '/home/sdwivedi/f16_propel/smpolicysynth/')

from synth.main.learn_conds import * 
from synth.main.learn_modes import * 
from synth.policy.prob_state_machine import * 
from synth.policy.state_machine import *
from synth.policy_grammars.straight_traj_grammar import *
from synth.policy_grammars.traj_opt_wrapper import *



def distill_into_sm(h, gym_env):
    gym_env.reset()
    nm_unroll = 8
    nm_sm = 2
    timesteps = 7
    cond_depth = 2

    pg  = StraightTrajGrammar(gym_env, nm_unroll, timesteps)
    opt = TrapOptWrapper(gym_env, pg, 1)

    opt.init_full_policy()

    trajs = []
    trajs.append(opt.get_policy()[0])

    curr_state = opt.init_states[0]
    for i in range(len(trajs[0].modes)):
        action = h.act(torch.as_tensor(curr_state, dtype=torch.float32))
        nxt_state, _, _, _ = gym_env.step(action)
        curr_state = nxt_state
        print(action)
        print(trajs[0].modes[i])
        for j in range(len(action)):
            trajs[0].modes[i][j][0] = action[j]

    actions_mean, actions_std, mode_mapping = learn_modes_n_mapping(gym_env, opt.init_states, trajs, [1], None, nm_sm)
    conds, conds_std = optimize_conds(gym_env, opt.init_states, trajs,  mode_mapping, nm_sm, [1], cond_depth)

    sm = ProbStateMachinePolicy(gym_env, actions_mean, actions_std, conds, conds_std)
    return sm


def train_via_drl(f, g, lambda_, gym_env):
    o, ep_ret, ep_len = gym_env.reset(), 0, 0
    odim = len(gym_env.observation_space.high)
    adim = len(gym_env.action_space.high)
    replay_buffer = ReplayBuffer(obs_dim=odim, act_dim=adim, size=32)
    for t in range(32):
        a1 = f.act(torch.as_tensor(o, dtype=torch.float32))
        a2 = g.get_action(o) 
        a1 = np.array(a1)
        a2 = np.array(a2)
        print(a1)
        print(a2)
        a = a1
        if(len(a2) != 0 ):
            a  = a + lambda_*a2
        o2, r, d, _ = gym_env.step(a)
        ep_ret += r
        ep_len += 1
        replay_buffer.store(o, a, r, o2, d)
    o = o2  
    batch = replay_buffer.sample_batch(32)
    pi_optimizer = Adam(f.pi.parameters(), lr=0.001)
    o = batch['obs']
    q_pi = f.q(o, f.pi(o))
    loss_pi = -q_pi.mean()
    loss_pi.backward()
    pi_optimizer.step()


# propel with jeevana's projection operator
def smpropel():
    # load the saved policy
    gym_env = env_fn()
    ddpg_model = MLPActorCritic(gym_env.observation_space, gym_env.action_space, hidden_sizes=(256,256), activation=nn.ReLU) 
    ddpg_model.load_state_dict(torch.load('/home/sdwivedi/f16_propel/ddpg_model.pth'))
    num_iterations = 10000
    lambda_ = 0.3 
    f = ddpg_model
    for i in range(num_iterations):
        # distill h into state machine policy
        g = distill_into_sm(f, gym_env)
        train_via_drl(f, g, lambda_, gym_env)
        if(num_iterations%10):
            g.save('f16_sm')
    # h = g + lambda_*f -> This operation happens inside train_via_drl method
    return g


min_sm = ProbStateMachinePolicy(env_fn(), [], [], [], [])
min_sm.read("f16_sm")
print(min_sm.evaluate(np.random.choice(10, 13)))
