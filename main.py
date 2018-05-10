import numpy as np
import tensorflow as tf
import collections
import logging
import contextlib
import warnings
import coloredlogs
import itertools

import envs
import approximators
from replay_buffer import ReplayBuffer

TD_Step = collections.namedtuple('TD_Step', [
    'observation', 'action', 'reward', 'next_observation'
    ])

create_env = envs.nchain_nonslip
create_vfun = approximators.MyApproximator

log = logging.getLogger(__name__)

def policy_thompson(vfuns):
    def _(observations):
        return np.argmax([v.predict(observations, dropout=True) for v in vfuns], axis=-1)
    return _


def continous_rollout(env, output):
    assert False
    terminated = True
    while True:
        if terminated:
            observation = env.reset()
        action = yield observation
        new_observation, reward, terminated, _ = env.step(action)
        output((observation, action, reward, new_observation))
        observation = new_observation


def bundle_envrolls(envrolls):
    assert False
    actions = [None] * len(envrolls)
    while True:
        observations = [e.send(a) for e,a in zip(envrolls, actions)]
        actions = yield observations


class InteractionEngine:
    def __init__(self, create_env, policy):
        assert False
        self.policy = policy
        self._env_bundle = bundle_envrolls([create_env() for _ in range(1000)])

    def generate_experience(self):
        observations = next(self._env_bundle)
        observations = self._env_bundle.send(policy(observations))
        yield from asd


def rollout_engine(policy):
    assert False
    while True:
        new_policy = yield experience
        if new_policy is not None:
            policy = new_policy
            continue

        experience = []
        rollout(env, experience.append)


def learn(vfuns, experience):
    observations = [[] for _ in vfuns]
    rewards = [[] for _ in vfuns]
    next_observations = [[] for _ in vfuns]

    # Bucketize experience on action
    for o,action,r,p in experience:
        observations[action].append(o)
        rewards[action].append(r)
        next_observations[action].append(p)

    # Each action has its own estimator, train each separatly
    for action, action_vfun in enumerate(vfuns):
        # Convert to numpy arrays
        observations = np.array(observations[action], ndmin=1)
        rewards = np.array(rewards[action], ndmin=1)
        next_observations = np.array(next_observations[action], ndmin=1)

        # Learn a TD0 step
        td_discount = 1
        next_predictions = action_vfun.predict(next_observations, dropout=False)
        td_targets = rewards + td_discount * next_predictions
        action_vfun.learn(observations, td_targets)
        # Note: The learning rate is set in AdamOptimizer used in approximators.py


@contextlib.contextmanager
def InitializedTFSession():
    """
    Context for a tensorflow session that is initalized and installed as
    default.
    """
    with tf.Session() as sess, sess.as_default():
        log.debug("Starting new tf-session with graph variables reset.")
        sess.run(tf.global_variables_initializer())
        yield sess


def rollout(env, policy):
    observation = env.reset()
    terminated = False
    while not terminated:
        action = policy([observation])[0]
        new_observation, reward, terminated, _ = env.step(action)
        yield TD_Step(observation, action, reward, new_observation)
        observation = new_observation


def run():
    env = create_env()
    replay_buffer = ReplayBuffer(30000)
    vfuns = [create_vfun() for _ in range(env.action_space.n)]

    with InitializedTFSession() as sess:
        for i in range(10000):
            while True:
                experience = [*rollout(env, policy_thompson(vfuns))]
                replay_buffer.extend(experience)
                if not replay_buffer.seeded:
                    log.debug(f"Filling buffer: {len(replay_buffer)}")
                    continue
                break
            learn(vfuns, replay_buffer.sample(10000))


def main():
    # Print all logs. In color.
    coloredlogs.install(
            level='DEBUG',
            fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
            )
    # End of setup
    run()


def main_werror():
    # Throw exceptions for warnings.
    warnings.filterwarnings("error")
    main()


if __name__ == '__main__':
    main()
