import argparse
import functools
from functools import partial

import jax
import jax.numpy as jnp
import gymnasium as gym
from jaxrl_m.dmc import DMCGym


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def none_or_str(value):
    if value == 'None':
        return None
    return value

def create_environments(env_name):
    """
    Create training and evaluation environments based on environment name.
    
    Args:
        env_name (str): Name of the environment
        max_episode_steps (int): Maximum steps per episode for training env
        
    Returns:
        tuple: (training_env, evaluation_env)
    """
    
    
    gym_envs = ["InvertedDoublePendulum-v5","Walker2d-v5","HalfCheetah-v5","Hopper-v5","Ant-v5","Humanoid-v5"]
    
        # Gymnasium environments  
    if env_name in gym_envs:
        env = gym.wrappers.RecordEpisodeStatistics(
            gym.make(env_name, max_episode_steps=1000)
        )
        eval_env = gym.wrappers.RecordEpisodeStatistics(
            gym.make(env_name, max_episode_steps=1000)
        )
    
    
    else : 
        
        domain, task = env_name.split("-")
        env = DMCGym(domain, task)
        eval_env = DMCGym(domain, task)
 
    return env, eval_env

def get_max_steps_for_env(env_name):
    """
    Get the appropriate max_steps for training based on environment name.
    
    Args:
        env_name (str): Name of the environment
        
    Returns:
        int: Maximum training steps for the environment
    """
    if env_name in ["Humanoid-v5", "HumanoidStandup-v5", "walk", "stand", "trot", "run"]:
        return 5_000_000
    elif env_name == "InvertedDoublePendulum-v5":
        return 200_000
    else:
        return 1_000_000  # default


def logp_from_pre_actions(apply_fn, params, obs, pre_a, tanh_squash=True):
    dist = apply_fn({'params': params}, obs)
    pre_log = dist.log_prob(pre_a)
    if tanh_squash:
        corr = jnp.sum(2 * (jnp.log(2.) - pre_a - jax.nn.softplus(-2. * pre_a)), axis=-1)
        return pre_log - corr
    return pre_log

def get_recent(transitions, K):
    n = transitions['observations'].shape[0]
    sl = slice(max(0, n-K), n)
    return jax.tree.map(lambda x: x[sl], transitions)

def tree_polyak(beta, target, source):
    return jax.tree.map(lambda t, s: beta * t + (1.0 - beta) * s, target, source)
