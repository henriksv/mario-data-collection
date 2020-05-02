#from deepQ_keras import Agent
from deep_DQN import Agent
import numpy as np
from utilities import plotLearning
import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from wrappers import wrapper
import json

import os
import cv2
from collections import deque
from statistics import mean as mean

def bvp_signal(state):
    return 1


def add_signal(signal, reward, weight):
        
    assert 0 <= weight <= 1 
    new_reward = reward
        
    return new_reward


if __name__=='__main__':
   
   # fname_base = 'mario_DDQN-20_60-64_32-8-4_32-4-2_32-3-1_512'
    fname_base = 'mario_DDQN_testingDQN3'
    output_dir = './' + fname_base
    fname_base = output_dir + '/' + fname_base

    if not os.path.exists(output_dir):
        print('not')
        os.makedirs(output_dir)
    
    #dims = (240, 256, 3)
    dims = (84, 84, 4)
    n_frames = 4
    img_rows, img_cols = dims[0], dims[1] #Downscaled

    memory_size = 500
    n_games = 100
    plot_check = 10
    model_save = 10
    print_check = 5
    episodes_save_check = 10

    fname_eval = fname_base + '_eval.h5'
    fname_next = fname_base + '_next.h5'

    episode_path = fname_base + '_episodes.json'

    agent = Agent(gamma=0.9, epsilon=1.0, alpha=0.00025, input_dims=dims, n_actions=7, mem_size=memory_size, 
                    batch_size=32, epsilon_end=0.01, replace=50, 
                    q_eval_fname=fname_eval, q_target_fname=fname_next)
    
    #env = gym.make('CartPole-v0')
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = wrapper(env)

    #agent.load_model()
    
    info = {}
    scores = []
    epsilon_history = []
    success_history = []
    episode_history = {}
    pos_history = []
    time_history = []
    progression_history = []

    
    i = 0
    for i in range(n_games):
    #while(True):
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

            #env.render()
            observation = observation_
            agent.learn()
            
    


        success = info['flag_get']
        position = info['x_pos']
        time = 400 - info['time']
        
        if info['flag_get']:
            print("Completed stage!")
 
        #episode_history.append([i, info['flag_get'], finish, score, avg_score, info['time'], agent.epsilon, info['x_pos']])

        epsilon_history.append(agent.epsilon)
        scores.append(score)
        success_history.append(success)
        pos_history.append(int(position))
        time_history.append(time)
        progression_rate = position / time
        progression_history.append(progression_rate)

       
            

        if i % print_check == 0:
            """    
            avg_success = np.mean(success_history[max(0,i-100):(i+1)])
            avg_dist    = np.mean(pos_history[max(0,i-100):(i+1)])
            avg_time    = np.mean(time_history[max(0,i-100):(i+1)])
            avg_score   = np.mean(scores[max(0,i-100):(i+1)])
            avg_progression = np.mean(progression_history[max(0, i-100):(i-1)])
            """    
            avg_success = mean(success_history[max(0,i-100):(i+1)])
            avg_dist    = mean(pos_history[max(0,i-100):(i+1)])
            avg_time    = mean(time_history[max(0,i-100):(i+1)])
            avg_score   = mean(scores[max(0,i-100):(i+1)])
            avg_progression = mean(progression_history[max(0, i-100):(i+1)])
            

            print('episode'+str(i)+':')
            print('score            %.2f' % score, '     average score %.2f' % avg_score)
            print('time             %.2f' % (400 - info['time']) ,'       average time %.2f' % avg_time)
            print('distance         %.2f' % info['x_pos'] , '     average distance %.2f' % avg_dist)
            print('success          %.2f' % info['flag_get'] ,  '        average success %.2f' % avg_success)
            print('progression_rate %.2f' % progression_rate, '       avg_progression %.2f' % avg_progression)
            
        if i % episodes_save_check:

            episode_history = {
                'epsilon' : epsilon_history,
                'scores' : scores,
                'success' : success_history,
                'position' : pos_history,
                'time' : time_history,
                'progression' : progression_history
            }

            with open(episode_path, 'w') as outfile:
                json.dump(episode_history, outfile)



        if (i+1) % model_save == 0 and i > 0:
            #agent.save_model(fname_base + '.h5')
            agent.save_models()
        
        if (i+1) % plot_check == 0 and i > 0:

            # for plotLearning
            #filename = fname_base + '_ep' + str(i+1) + '.png'
            filename = fname_base + '.png'
            
            x = [j+1 for j in range(i+1)]
            plotLearning(x, scores, epsilon_history, filename)

        #i +=1