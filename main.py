import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import envs
import approximators

create_env = envs.nchain_nonslip
create_vfun = approximators.MyApproximator


def policy_thompson(vfuns):
    def _(observation):
        return np.argmax([v.predict([observation], dropout=True) for v in vfuns])
    return _


def rollout(env, policy):
    # Rollout a single episode
    experience = []
    terminated = False
    observation = env.reset()
    while not terminated:
        action = policy(observation)
        new_observation, reward, terminated, _ = env.step(action)
        experience.append((observation, action, reward, new_observation))
        observation = new_observation
    return experience


def learn(vfuns, experience):
    os = [[] for _ in vfuns]
    rs = [[] for _ in vfuns]
    ps = [[] for _ in vfuns]
    for o,a,r,p in experience:
        os[a].append(o)
        rs[a].append(r)
        ps[a].append(p)
    for a, vfun in enumerate(vfuns):
        osa = np.array(os[a])
        rsa = np.array(rs[a])
        psa = np.array(ps[a])
        d = 0.9 # discount
        # FIXME: this dropout should be false
        vfun.learn(osa, rsa + d*vfun.predict(psa, dropout=True))




def main():
    plt.ion()
    fig = plt.figure()
    ax = fig.gca()
    env = create_env()
    action_count = env.action_space.n
    vfuns = [create_vfun() for _ in range(action_count)]
    with tf.Session() as sess, sess.as_default():
        sess.run(tf.global_variables_initializer())

        # Interaction loop
        for i in range(10000):
            experience = rollout(env, policy_thompson(vfuns))
            learn(vfuns, experience)
            if i % 1 == 0:
                reward_sum = sum(x[1] for x in experience)
                print(f"{i}: reward sum {reward_sum}")
                distributions = {a: vfun.predict([0]*100000, dropout=True) for a, vfun in enumerate(vfuns)}
                ax.clear()
                pd.DataFrame( distributions ).plot.hist(ax=ax, histtype='step', density=True, bins=100, alpha=0.8)
                fig.tight_layout()
                fig.canvas.draw()


if __name__ == '__main__':
    main()
