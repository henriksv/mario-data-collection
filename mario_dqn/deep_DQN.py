# URL: https://www.youtube.com/watch?v=5fHngyN8Qhw
# Also: https://www.statworx.com/at/blog/using-reinforcement-learning-to-play-super-mario-bros-on-nes-using-tensorflow/

from keras.layers import Dense, Activation, Conv2D, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import Adam

import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.mem_cntr = 0

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]


        #print('actions', actions.shape)

        return states, actions, rewards, states_, terminal

def build_dqn_old(lr, n_actions, input_dims, fc1_dims, fc2_dims):

    model = Sequential([
        Dense(fc1_dims, input_shape=(input_dims, )),
        Activation('relu'),
        Dense(fc2_dims),
        Activation('relu'),
        Dense(n_actions)
    ])

    model.compile(optimizer=Adam(lr=lr), loss='mse')

    return model

def build_dqn(lr, n_actions, input_dims):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=8, strides=4, activation='relu',
                     input_shape=(*input_dims,)))

    model.add(Conv2D(filters=32, kernel_size=4, strides=2, activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(n_actions))

    model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error')

    return model

class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, input_dims, replace, 
                epsilon_dec=0.9999, epsilon_end=0.01, mem_size=1000000, 
                q_eval_fname='q_eval.h5', q_target_fname='q_target.h5'):
        
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.replace = replace
        self.learn_step = 0
        self.q_target_model_file = q_target_fname
        self.q_eval_model_file = q_eval_fname

        self.train_every = 3 # Train on every .th step

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)
        self.q_eval = build_dqn(alpha, n_actions, input_dims)#, 256, 256)
        self.q_next = build_dqn(alpha, n_actions, input_dims)#, 256, 256)

    def replace_target_network(self):
        if self.replace is not None and self.learn_step % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if  self.memory.mem_cntr < self.batch_size:
            return
        
        # Train on every
        if self.memory.mem_cntr == 0 or self.memory.mem_cntr % self.train_every != 0:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        self.replace_target_network()

        q_eval = self.q_eval.predict(state)
        q_next = self.q_next.predict(new_state)


        # Revert back from one-hot encoding
        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)

        q_target = q_eval[:]
        #q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, action_indices] = \
            reward + self.gamma*np.max(q_next, axis=1)*done

        _= self.q_eval.fit(state, q_target, verbose=0)
        # self.q_eval.train_on_batch(state, q_target)

        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon*self.epsilon_dec
        else:
            self.epsilon = self.epsilon_min
        
        self.learn_step += 1


    def save_models(self):
        self.q_eval.save(self.q_eval_model_file)
        self.q_next.save(self.q_target_model_file)
        print('... saving models ...')

    def load_models(self):
        self.q_eval = load_model(self.q_eval_model_file)
        self.q_nexdt = load_model(self.q_target_model_file)
        print('... loading models ...')