import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import backgammon2
from replay_buffer import ReplayBuffer


# 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
# 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152,
# 4194304, 8388608, 16777216, 33554432, 67108864, 134217728,
# 268435456, 536870912, 1073741824, 2147483648

CONFIG = {
    "input_shape": 29,
    "epsilon": 0.05,
    "learning_rate": 0.05,
    "gamma": 0.99,
    "batch_size": 256,
    "train_model_steps": 512,
    "update_target_steps": 32768,
    "save_model_steps": 262144,
    "saved_model_dir": "teddy_weights2",
    "saved_model": None,
    # "saved_model": "./teddy_weights/...",
    "saved_model_race": None,
    # "saved_model_race": "./teddy_weights/...",
}


if CONFIG["saved_model"] is not None:
    DQN = tf.keras.models.load_model(CONFIG["saved_model"])
else:

    # Deep Q-network
    DQN = keras.Sequential(
        [
            layers.Dense(
                50,
                activation="relu",
                kernel_initializer="random_uniform",
                input_shape=(CONFIG["input_shape"],),
            ),
            layers.Dense(50, activation="relu", kernel_initializer="random_uniform",),
            layers.Dense(4, activation="relu", kernel_initializer="random_uniform"),
            layers.Dense(1, activation="sigmoid", kernel_initializer="random_uniform"),
        ]
    )
    DQN.compile(optimizer="Adam", loss="mse")

# Target network to stabilize
# https://www.tensorflow.org/api_docs/python/tf/keras/models/clone_model
DQN_target = tf.keras.models.clone_model(DQN)
DQN_target.compile(optimizer="Adam", loss="mse")


if CONFIG["saved_model_race"] is not None:
    DQN = tf.keras.models.load_model(CONFIG["saved_model"])
else:
    # Deep Q-network for race mode
    DQN_race = tf.keras.models.clone_model(DQN)
    DQN_race.compile(optimizer="Adam", loss="mse")

# Target network to stabilize
DQN_race_target = tf.keras.models.clone_model(DQN)
DQN_race_target.compile(optimizer="Adam", loss="mse")

# replay buffer to reduce correlation
B = ReplayBuffer(obs_dim=29, size=2048, batch_size=256)
B_race = ReplayBuffer(obs_dim=29, size=2048, batch_size=256)

# for tracking progress
counter = 0
race_counter = 0
saved_models = []

print("Network architecture: \n", DQN)

rand_buffer = list()


def get_rand(buffer_size=100_000):
    global rand_buffer
    try:
        return rand_buffer.pop()
    except IndexError:
        rand_buffer = list(np.random.uniform(size=buffer_size))
        return rand_buffer.pop()


def action(board, board_array, train=False, **kwargs):

    global counter
    global race_counter

    is_race = backgammon2.is_race(board)

    if not is_race:
        model = DQN
        model_name = "DQN"
        target_model = DQN_target
        buffer = B
        counter += 1
        cnt = counter
    else:
        model = DQN_race
        model_name = "DQN_race"
        target_model = DQN_race_target
        buffer = B_race
        race_counter += 1
        cnt = race_counter

    values = model(board_array)
    action = np.argmax(values)

    if train:

        # with some probability just pick a random move
        rand = get_rand()
        if rand < CONFIG["epsilon"]:
            action = np.random.randint(len(board_array))

        next_board = board_array[action]
        buffer.store(obs=board, next_obs=next_board)

        if (cnt % CONFIG["train_model_steps"] == 0) and (cnt != 0):
            # print(f"Training model - {cnt}")

            batch = buffer.sample_batch()

            Q = model(batch["obs"])
            Q_prime = target_model(batch["next_obs"])
            reward = backgammon2.game_over_array(batch["next_obs"])

            target = Q + CONFIG["learning_rate"] * (
                reward + CONFIG["gamma"] * Q_prime - Q
            )
            # ^ is this right???

            model.train_on_batch(batch["obs"], target)

        if cnt % CONFIG["update_target_steps"] == 0:
            # print(f"Updating target model - {cnt}")
            target_model.set_weights(model.get_weights())

        if cnt % CONFIG["save_model_steps"] == 0:
            print(f"Saving model - {cnt}")
            filepath = f"./{CONFIG['saved_model_dir']}/{model_name}_{str(cnt)}"
            print("saving weights in file:", filepath)
            model.save(filepath, overwrite=True, include_optimizer=True)

    return action
