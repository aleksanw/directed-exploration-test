import gym as openai
import tensorflow as tf
numeric_feature = tf.feature_column.numeric_column


def create_qfun(observation_shape):
    # Three hidden layers with 128 neurons. Separate network for each action.
    qnetwork_shape = [128, 128, 128]

    feature_info = [
        numeric_feature(
            key='observation',
            shape=observation_shape,
        )]
    qfun = tf.estimator.DNNRegressor(
        feature_columns=feature_info,
        hidden_units=qnetwork_shape,
        dropout=0.25, # NOTE
        )
    return qfun


def train(estimator, observations, rewards):
    estimator.train(lambda: tf.data.Dataset.from_tensors(
        ({'observation': observations}, rewards)
        ))


def predict(estimator, observations):
    return [*estimator.train(lambda: tf.data.Dataset.from_tensors(
        {'observation': observations}
        ))]


def create_env():
    # Non-slip NChain
    env = openai.make('NChain-v0')
    env.unwrapped.slip = 0
    return env


def thompson_action(qfuns, observation):
    return np.argmax([predict(qfun, [observation])[0] for qfun in qfuns])


def main():
    with tf.Session():
        env = create_env()
        action_count = env.action_space.n
        qfuns = [create_qfun() for _ in range(action_count)]

        # Interaction loop
        while True:
            observations = [[] for _ in range(action_count)]
            rewards = [[] for _ in range(action_count)]

            # Run an episode
            observation = env.reset()
            episode_finished = False
            while not episode_finished:
                action = thompson_action(qfuns, observation)
                observations[action].append(observation)
                observation, reward, episode_finished, _ = env.step(action)
                rewards[action].append(reward)

            # Train qfuns
            for qf, acts, rews in zip(qfuns, observation, rewards):
                train(qf, acts, rews)


if __name__ == '__main__':
    main()
