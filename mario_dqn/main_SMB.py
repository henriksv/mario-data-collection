#from deepQ_keras import Agent
from deep_DQN import Agent
import numpy as np
from utils import plotLearning
import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from wrappers import wrapper

import cv2
from collections import deque


if __name__=='__main__':
    #env = gym.make('CartPole-v0')
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = wrapper(env)

   # fname_base = 'mario_DDQN-20_60-64_32-8-4_32-4-2_32-3-1_512'
    fname_base = './mario_DDQN_3lives'
    n_games = 10000

    #dims = (240, 256, 3)
    dims = (84, 84, 4)
    n_frames = 4
    img_rows, img_cols = dims[0], dims[1] #Downscaled

    fname_eval = fname_base + '_eval.h5'
    fname_next = fname_base + '_next.h5'

    agent = Agent(gamma=0.9, epsilon=1.0, alpha=0.00025, input_dims=dims, n_actions=7, mem_size=100000, 
                    batch_size=32, epsilon_end=0.01, replace=50, 
                    q_eval_fname=fname_eval, q_target_fname=fname_next)

    #agent.load_model()

    scores = []
    eps_history = []

    #for i in range(n_games):
    i = 0
    while(True):
        done = False
        score = 0
        observation = env.reset()
        #print(observation.shape)
        #observation = cv2.resize(observation, dsize=(img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        #print(observation.shape)

        #nstep = 0

        while not done:# and nstep < 100:
            #print('step: ', nstep)
            #nstep+=1

            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)

            #observation_ = cv2.resize(observation_, dsize=(img_cols, img_rows), interpolation=cv2.INTER_CUBIC)

            score += reward
            agent.remember(observation, action, reward, observation_, done)

            env.render()
            observation = observation_
            agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[max(0,i-100):(i+1)])
        print('episode ', i, 'score %.2f' % score, ' average score %.2f' % avg_score)

        if i % 100 == 0 and i > 0:
            #agent.save_model(fname_base + '.h5')
            
            
            agent.save_models()
        
        if i % 1000 == 0 and i > 0:

            # for plotLearning
            #filename = fname_base + '_ep' + str(i+1) + '.png'
            filename = fname_base + '.png'
            
            x = [j+1 for j in range(i+1)]
            plotLearning(x, scores, eps_history, filename)

        i +=1