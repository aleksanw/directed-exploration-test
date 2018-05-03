import gym

def nchain_nonslip():
    # Non-slip NChain
    gymenv = gym.make('NChain-v0')
    gymenv.unwrapped.slip = 0
    gymenv.unwrapped.n = 40
    return gymenv
