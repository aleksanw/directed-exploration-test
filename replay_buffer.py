import collections
import logging
import random


log = logging.getLogger(__name__)


class ReplayBuffer(collections.deque):
    """Stores oaro-tuples. This in essence creates a transition model that will
    be sampled for learning."""

    def __init__(self, size):
        super().__init__(self, maxlen=size)

    @property
    def seeded(self):
        return len(self) == self.maxlen

    def sample(self, sample_size):
        if len(self) != self.maxlen:
            log.warning("Replay buffer was sampled from before fully seeded. "
                        "Ensure `replay_buffer.seeded` is `True` before "
                        "calling `replaybuffer.sample`.")
        return random.choices(self, k=sample_size)
