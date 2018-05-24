import numpy as np
import tensorflow as tf
import collections
import logging
import coloredlogs
import itertools
import os
from pprint import pformat

import tensor_wrapper as tw
import policy_thompson
import plot
import envs
import approximators
from replay_buffer import ReplayBuffer

TD_Step = collections.namedtuple('TD_Step', [
    'observation', 'action', 'reward', 'next_observation'
    ])

create_env = envs.nchain_nonslip
create_vfun = approximators.MyApproximator

log = logging.getLogger(__name__)


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
        next_predictions = action_vfun.predict_reward(next_observations, dropout=False)
        td_targets = rewards + td_discount * next_predictions
        action_vfun.learn(observations, td_targets)
        # Note: The learning rate is set in AdamOptimizer used in approximators.py


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

    with tw.InitializedSession() as sess:
        for i in itertools.count():
            plot.distribution(
                    f"imgs/state0-{i}.png",
                    "Action-reward (discounted) distribution in state 0",
                    "Reward",
                    {
                        "Action 0": [vfuns[0].predict_reward([0]*10000, dropout=True)],
                        "Action 1": [vfuns[1].predict_reward([0]*10000, dropout=True)],
                    },
                    )

            experience = [*rollout(env, policy_thompson.create([lambda x: v.predict_reward(x, dropout=True) for v in vfuns]))]
            log.debug(f"Experience gained: {pformat(experience)}")
            replay_buffer.extend(experience)
            if not replay_buffer.seeded:
                log.debug(f"Filling buffer: {len(replay_buffer)}")
                continue
            learn(vfuns, replay_buffer.sample(10000))


def main():
    # Print all logs. In color.
    coloredlogs.install(
            level='DEBUG',
            fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
            )
    os.makedirs('imgs', exist_ok=True)
    # End of setup
    run()


if __name__ == '__main__':
    main()
