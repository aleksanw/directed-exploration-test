import logging
log = logging.getLogger(__name__)

import contextlib
import tensorflow as tf

@contextlib.contextmanager
def InitializedSession():
    """
    Context for a tensorflow session that is initalized and installed as
    default.
    """
    with tf.Session() as sess, sess.as_default():
        log.debug("Starting new tf-session with graph variables reset.")
        sess.run(tf.global_variables_initializer())
        yield sess


