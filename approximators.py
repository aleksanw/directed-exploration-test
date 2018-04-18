import tensorflow as tf

class MyApproximator:
    def __init__(self):
        # Extend default TF-graph.
        # TF-session is not needed at this point. TF-globals must be initalized by
        # caller before calling predict/learn.
        # Observation is scalar. Hopefully this works for NChain.
        self._observations = tf.placeholder(tf.float32, shape=[None])
        head = self._observations
        # Network from double-uncertain paper. Separate network for each
        # action. Three dense relu layers of 128. Dropout with p-keep=75%
        # enabled during prediction.
        head = tf.reshape(head, [-1, 1])
        for _ in range(3):
            head = tf.layers.dense(head, 128)
            # Parameter `training` determines if dropout is enabled.
            head = tf.layers.dropout(head, rate=0.25, training=True)
        head = tf.layers.dense(head, 1)
        head = tf.reshape(head, [-1])
        self._value_predictions = head

        self._value_targets = tf.placeholder(tf.float32, shape=[None])
        self._learn_op = tf.train.AdamOptimizer().minimize(
            tf.losses.mean_squared_error(self._value_targets, self._value_predictions)
            )

    def learn(self, observations, values):
        # Must be called in a TF-session context.
        tf.get_default_session().run(self._learn_op, feed_dict={
            self._observations: observations,
            self._value_targets: values,
            })

    def predict(self, observations, dropout=False):
        # Without dropout is not yet implemented.
        assert dropout == True
        # Must be called in a TF-session context.
        return self._value_predictions.eval(feed_dict={
            self._observations: observations,
            })
