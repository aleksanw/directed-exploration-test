import logging
import numpy as np

log = logging.getLogger(__name__)

def create(reward_distribution_samplers):
    """
        reward_distribution_samplers :: [obs] -> [rew], obs :: [float]
        returns :: [act]
    """
    def policy(observations):
        # Indulgent thompson would start with something like this:
        #observations = observations.repeat(oversample_level, axis=0)
        samples = np.array([f(observations) for f in reward_distribution_samplers])
        selected_action = np.argmax(samples, axis=0)
        #log.debug(f"action {selected_action} samples {[*samples.flatten()]}")
        return selected_action
    return policy
