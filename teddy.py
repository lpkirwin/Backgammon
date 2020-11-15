from copy import deepcopy

import numpy as np
import tensorflow as tf

from fancy_buffers import PrioritizedReplayBuffer, NStepBuffer, GameBuffer

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
    "learning_rate": 0.1,  # applied when generating target
    "final_learning_rate": 0.01,
    "anneal_learning_rate_steps": 16777216,
    "gamma": 0.999,  # discount factor
    "lambda_": 0.999,  # time decay
    "epsilon": 0.2,  # chance of random action
    "final_epsilon": 0.02,
    "anneal_epsilon_steps": 16777216,
    "alpha": 0.8,  # amount of prioritisation (higher is more)
    "beta": 0.4,  # amount of importance sampling (higher is more unbiased?)
    "final_beta": 1.0,
    "anneal_beta_steps": 8388608,
    "temperature": 1,  # for choice randomisation
    "final_temperature": 0.02,
    "anneal_temperature_steps": 16777216,
    "buffer_size": 512,  # number of moves to keep in memory
    "buffer_n_step": 2,  # ~ how far back to apply rewards
    "batch_size": 5,
    "train_model_steps": 16,
    "update_target_steps": 32768,
    "save_model_steps": 262144,
    "save_model_dir": "teddy_models",
    "saved_model": None,
    "name": "v0",
    "is_trainable": True,
    "is_saveable": True,
    "rollouts_during_training": False,
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
        self.parent = None

        # load model, or make a new one
        if self.saved_model is not None:
            path = f"./{self.save_model_dir}/{self.saved_model}"
            print("Loading model from", path)
            self.model = tf.keras.models.load_model(path)
            self.target_model = tf.keras.models.load_model(path)
            self.counter = load_counter(path)
        else:
            print("Building model from scratch")
            self.model = make_model(self.input_shape)
            self.target_model = tf.keras.models.clone_model(self.model)
            self.target_model.compile(optimizer="Adam", loss="mse")
            self.target_model.set_weights(self.model.get_weights())
            self.counter = 0

        # # load a buffer (either n_step or simple)
        # if self.buffer_n_step > 2:
        #     self.buffer = NStepBuffer(
        #         obs_dim=self.input_shape,
        #         size=self.buffer_size,
        #         batch_size=self.batch_size,
        #         # alpha=self.alpha,
        #         n_step=self.buffer_n_step,
        #         gamma=self.gamma,
        #     )
        # else:
        #     self.buffer = ReplayBuffer(
        #         obs_dim=self.input_shape,
        #         size=self.buffer_size,
        #         batch_size=self.batch_size,
        #     )
        self.buffer = GameBuffer(
            obs_dim=self.input_shape,
            max_games=self.buffer_size,
            batch_size=self.batch_size,
        )

        if self.rollouts_during_training:
            self.get_q = self.model_values_rollout
            self.get_q_prime = self.target_values_rollout
        else:
            self.get_q = self.model_values
            self.get_q_prime = self.target_values

        # # load a rollout manager
        # self.rollouts = RolloutManager(data_size=500_000)

    def game_over_update(self, board, won):
        reward = 1 if won else -1
        state = backgammon2.board_to_state(board)
        self.buffer.store(
            obs=state, act=0, next_obs=state, rew=reward, done=True,
        )
        if self.is_trainable:
            self.game_train()

    @tf.function(experimental_relax_shapes=True)
    def model_values(self, x):
        return self.model(x)

    @tf.function(experimental_relax_shapes=True)
    def target_values(self, x):
        return self.target_model(x)

    def model_values_rollout(self, x):
        return self._rollout_values(x, self.model_values)

    def target_values_rollout(self, x):
        return self._rollout_values(x, self.target_values)

    def _rollout_values(self, board_array, value_func):
        rollout_boards = self.rollouts.full_rollout_boards(board_array)
        rollout_states = backgammon2.board_array_to_state_array(rollout_boards)
        rollout_values = np.array(value_func(rollout_states))
        rollout_values = rollout_values.astype(np.float32).flatten()
        board_values = self.rollouts.maxmin_rollout_value(rollout_values)
        return board_values.reshape(-1, 1)

    def _noisy_choice(self, values):
        values = np.array(values).flatten()
        try:
            k = np.e ** (values / self.temperature)
            prb = k / k.sum()
        except Exception as e:
            print(e)
            return np.argmax(values)
        return np.random.choice(len(prb), p=prb)

    def action(
        self,
        board,
        board_array,
        train=False,
        rollout=False,
        print_value=False,
        **kwargs,
    ):

        # state = backgammon2.board_to_state(board)
        state_array = backgammon2.board_array_to_state_array(board_array)

        if rollout:
            values = self.model_values_rollout(state_array)
        else:
            values = self.model_values(state_array)

        if train and self.is_trainable:

            self.counter += 1

            # # with some probability just pick a random move
            # rand = get_rand()
            # if rand < self.epsilon:
            #     action = np.random.randint(len(board_array))

            best_action = np.argmax(values)
            action = self._noisy_choice(values)

            # next_board = board_array[action]
            best_state = state_array[best_action]
            next_state = state_array[action]

            self.buffer.store(
                # obs=state, act=action, next_obs=next_state, rew=0, done=False
                obs=next_state,
                act=action,
                next_obs=best_state,
                rew=0,
                done=False,
            )

            # if ((self.counter % self.train_model_steps) == 0) & (
            #     len(self.buffer) > self.batch_size
            # ):

            #     self.train()
            #     self.anneal()

            if (self.counter % self.update_target_steps) == 0:
                print(f"Updating target model - {self.counter}")
                self.target_model.set_weights(self.model.get_weights())

            if (self.counter % self.save_model_steps) == 0:
                if self.is_saveable:
                    self.save_model()

        else:
            action = np.argmax(values)

        if print_value:
            print("Max value is:", np.max(values))

        return action

    def train(self):

        batch = self.buffer.sample_batch(beta=self.beta)

        Q = self.get_q(batch["obs"])
        Q_prime = self.get_q_prime(batch["next_obs"])

        reward = batch["rews"].reshape(-1, 1)
        done = batch["done"].reshape(-1, 1)

        # if self.buffer_n_step > 1:
        #     weights = batch["weights"].reshape(-1)
        #     indices = batch["indices"]
        # else:
        #     weights = None

        # if done, target = reward, else use Q learning expression
        target = done * reward + (1 - done) * (
            Q + self.learning_rate * (reward + self.gamma * Q_prime - Q)
        )

        y = np.array(target)

        # self.model.train_on_batch(batch["obs"], y, sample_weight=weights)
        self.model.train_on_batch(batch["obs"], y)

        # # update priorities
        # if self.buffer_n_step > 1:
        #     new_priorities = (Q - target) ** 2 + 0.001
        #     self.buffer.update_priorities(indices, new_priorities)

    def game_train(self):

        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample_batch()

        Q = self.get_q(batch["obs"])
        Q_target = self.get_q_prime(batch["obs"])

        reward = batch["rews"].reshape(-1, 1)
        done = batch["done"].reshape(-1, 1)

        Q_prime = np.zeros(shape=len(Q_target), dtype=np.float32)
        for idx in range(len(Q) - 1, -1, -1):

            if done[idx]:
                val = reward[idx]
            else:
                val = Q_target[idx + 1]
                # q1 = Q_target[idx + 1]
                # qp1 = Q_prime[idx + 1]
                # val = q1 + self.lambda_ * (qp1 - q1)
            Q_prime[idx] = val

        Q_prime = Q_prime.reshape(-1, 1)

        # if done, target = reward, else use Q learning expression
        target = done * reward + (1 - done) * (
            Q + self.learning_rate * (reward + self.gamma * Q_prime - Q)
        )

        y = np.array(target)
        self.model.train_on_batch(batch["obs"], y)

        self.anneal()

    def anneal(self):

        # annealing beta
        fraction = min(self.counter / self.anneal_beta_steps, 1.0)
        self.beta += fraction * (self.final_beta - self.beta)

        # annealing epsilon
        fraction = min(self.counter / self.anneal_epsilon_steps, 1.0)
        self.epsilon += fraction * (self.final_epsilon - self.epsilon)

        # annealing learning rate
        fraction = min(self.counter / self.anneal_learning_rate_steps, 1.0)
        self.learning_rate += fraction * (self.final_learning_rate - self.learning_rate)

        # annealing temperature
        fraction = min(self.counter / self.anneal_temperature_steps, 1.0)
        self.temperature += fraction * (self.final_temperature - self.temperature)

    def make_copy(self, **kwargs):
        # can't do a simple deepcopy with tensorflow (I think),
        # so just initialise a new object
        config = deepcopy(self.config)
        config.update(**kwargs)
        new = AgentTeddy(**config)
        # set attrs that may have changed
        for attr in ["counter", "beta", "epsilon"]:
            setattr(new, attr, getattr(self, attr))
        new.model.set_weights(self.model.get_weights())
        new.target_model.set_weights(self.target_model.get_weights())
        new.parent = self.name
        return new

    def save_model(self, filename=None):
        filename = filename or self.name
        print(f"Saving model - {self.counter}")
        filepath = f"./{self.save_model_dir}/{filename}"
        print("saving weights in file:", filepath)
        self.model.save(filepath, overwrite=True, include_optimizer=True)
        save_counter(filepath, self.counter)

    def __str__(self):
        return f"teddy_{self.name}_{self.parent}_{self.counter}"
