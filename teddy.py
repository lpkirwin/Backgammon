from copy import deepcopy

import numpy as np
import tensorflow as tf

from fancy_buffers import PrioritizedReplayBuffer

import backgammon2
from replay_buffer import ReplayBuffer
from rollouts import RolloutManager

# powers of 2:
# 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
# 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152,
# 4194304, 8388608, 16777216, 33554432, 67108864, 134217728,
# 268435456, 536870912, 1073741824, 2147483648

CONFIG = {
    "input_shape": 79,
    "learning_rate": 0.15,  # applied when generating target
    "final_learning_rate": 0.01,
    "anneal_learning_rate_steps": 16777216,
    "gamma": 0.99,  # discount factor
    "epsilon": 0.2,  # chance of random action
    "final_epsilon": 0.02,
    "anneal_epsilon_steps": 16777216,
    "alpha": 0.8,  # amount of prioritisation (higher is more)
    "beta": 0.4,  # amount of importance sampling (higher is more unbiased?)
    "final_beta": 1.0,
    "anneal_beta_steps": 16777216,
    "buffer_size": 4096,  # number of moves to keep in memory
    "buffer_n_step": 1,  # ~ how far back to apply rewards
    "batch_size": 32,
    "train_model_steps": 32,
    "update_target_steps": 32768,
    "save_model_steps": 262144,
    "save_model_dir": "teddy_models",
    "saved_model": None,
    "model_name": "teddy_v0",
    "is_trainable": True,
}


rand_buffer = list()


# This makes makes things a bit faster by pre-generating
# lots of random values and then popping them from a list
# as needed
def get_rand(buffer_size=100_000):
    global rand_buffer
    try:
        return rand_buffer.pop()
    except IndexError:
        rand_buffer = list(np.random.uniform(size=buffer_size))
        return rand_buffer.pop()


def save_counter(filepath, counter):
    with open(filepath + ".counter", "w") as file:
        file.write(str(counter))


def load_counter(filepath):
    with open(filepath + ".counter", "r") as file:
        counter = int(file.read())
    return counter


def make_model(input_shape):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                50,
                activation="relu",
                kernel_initializer="random_uniform",
                input_shape=(input_shape,),
            ),
            tf.keras.layers.Dense(
                50, activation="relu", kernel_initializer="random_uniform",
            ),
            tf.keras.layers.Dense(
                4, activation="relu", kernel_initializer="random_uniform"
            ),
            tf.keras.layers.Dense(
                1, activation=None, kernel_initializer="random_uniform"
            ),
        ]
    )
    model.compile(optimizer="Adam", loss="mse")
    return model


class AgentTeddy(object):
    def __init__(self, **kwargs):

        # update and set attrs from config
        self.config = deepcopy(CONFIG)
        for k, v in kwargs.items():
            self.config[k] = v
        for k, v in self.config.items():
            setattr(self, k, v)

        # load model, or make a new one
        if self.saved_model is not None:
            print("Loading model from", self.saved_model)
            self.model = tf.keras.models.load_model(self.saved_model)
            self.target_model = tf.keras.models.load_model(self.saved_model)
            self.counter = load_counter(self.saved_model)
        else:
            print("Building model from scratch")
            self.model = make_model(self.input_shape)
            self.target_model = tf.keras.models.clone_model(self.model)
            self.target_model.compile(optimizer="Adam", loss="mse")
            self.target_model.set_weights(self.model.get_weights())
            self.counter = 0

        # load a buffer (either n_step or simple)
        if self.buffer_n_step > 1:
            self.buffer = PrioritizedReplayBuffer(
                obs_dim=self.input_shape,
                size=self.buffer_size,
                batch_size=self.batch_size,
                alpha=self.alpha,
                n_step=self.buffer_n_step,
                gamma=self.gamma,
            )
        else:
            self.buffer = ReplayBuffer(
                obs_dim=self.input_shape,
                size=self.buffer_size,
                batch_size=self.batch_size,
            )

        # load a rollout manager
        self.rollouts = RolloutManager(data_size=500_000)

    def game_over_update(self, board, won):
        reward = 1 if won else -1
        state = backgammon2.board_to_state(board)
        self.buffer.store(
            obs=state, act=0, next_obs=state, rew=reward, done=True,
        )

    @tf.function(experimental_relax_shapes=True)
    def model_values(self, x):
        return self.model(x)

    @tf.function(experimental_relax_shapes=True)
    def target_values(self, x):
        return self.target_model(x)

    def rollout_values(self, board_array, value_func):
        rollout_boards = self.rollouts.full_rollout_boards(board_array)
        rollout_states = backgammon2.board_array_to_state_array(rollout_boards)
        rollout_values = np.array(value_func(rollout_states))
        rollout_values = rollout_values.astype(np.float32).flatten()
        board_values = self.rollouts.maxmin_rollout_value(rollout_values)
        return board_values

    def action(self, board, board_array, train=False, rollout=False, **kwargs):

        state = backgammon2.board_to_state(board)
        state_array = backgammon2.board_array_to_state_array(board_array)

        if rollout:
            values = self.rollout_values(board_array, self.model_values)
        else:
            values = self.model_values(state_array)

        action = np.argmax(values)

        if train and self.is_trainable:

            self.counter += 1

            # with some probability just pick a random move
            rand = get_rand()
            if rand < self.epsilon:
                action = np.random.randint(len(board_array))

            # next_board = board_array[action]
            next_state = state_array[action]

            self.buffer.store(
                obs=state, act=action, next_obs=next_state, rew=0, done=False
            )

            if ((self.counter % self.train_model_steps) == 0) & (
                len(self.buffer) > self.batch_size
            ):

                batch = self.buffer.sample_batch(beta=self.beta)

                Q = self.model(batch["obs"])
                # Q_prime = self.target_model(batch["next_obs"])
                Q_prime = (
                    self.rollout_values(batch["next_obs"], self.target_values)
                ).reshape(-1, 1)

                reward = batch["rews"].reshape(-1, 1)
                done = batch["done"].reshape(-1, 1)

                if self.buffer_n_step > 1:
                    weights = batch["weights"].reshape(-1)
                    indices = batch["indices"]
                else:
                    weights = None

                # if done, target = reward, else use Q learning expression
                target = done * reward + (1 - done) * (
                    Q + self.learning_rate * (reward + self.gamma * Q_prime - Q)
                )

                y = np.array(target)

                self.model.train_on_batch(batch["obs"], y, sample_weight=weights)

                # update priorities
                if self.buffer_n_step > 1:
                    new_priorities = (Q - target) ** 2 + 0.001
                    self.buffer.update_priorities(indices, new_priorities)

                # annealing beta
                fraction = min(self.counter / self.anneal_beta_steps, 1.0)
                self.beta += fraction * (self.final_beta - self.beta)

                # annealing epsilon
                fraction = min(self.counter / self.anneal_epsilon_steps, 1.0)
                self.epsilon += fraction * (self.final_epsilon - self.epsilon)

                # annealing learning rate
                fraction = min(self.counter / self.anneal_learning_rate_steps, 1.0)
                self.learning_rate += fraction * (
                    self.final_learning_rate - self.learning_rate
                )

            if (self.counter % self.update_target_steps) == 0:
                print(f"Updating target model - {self.counter}")
                self.target_model.set_weights(self.model.get_weights())

            if (self.counter % self.save_model_steps) == 0:
                self.save_model()

        return action

    def make_copy(self):
        # can't do a simple deepcopy with tensorflow (I think),
        # so just initialise a new object
        new = AgentTeddy(**self.config)
        # set attrs that may have changed
        for attr in ["counter", "beta", "epsilon"]:
            setattr(new, attr, getattr(self, attr))
        new.model.set_weights(self.model.get_weights())
        new.target_model.set_weights(self.target_model.get_weights())
        return new

    def save_model(self):
        print(f"Saving model - {self.counter}")
        filepath = f"./{self.save_model_dir}/{self.model_name}"
        print("saving weights in file:", filepath)
        self.model.save(filepath, overwrite=True, include_optimizer=True)
        save_counter(filepath, self.counter)

    def __str__(self):
        return f"{self.model_name}_{self.counter}"
