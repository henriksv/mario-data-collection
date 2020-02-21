
import gym_super_mario_bros
import numpy as np
import gym

import json
from pyglet import clock
import time

import numpy as np
import cv2
import matplotlib.pyplot as plt


stage_order = [
    (1,1),
    (1,2),
    (1,3),
    (2,2),
    (1,4),
    (3,1),
    (4,1),
    (2,1),
    (2,3),
    (2,4),
    (3,2),
    (3,3),
    (3,4),
    (4,2)
]

#play_time = 600 # 10min
#play_time = 2100 # 35min

stage_order_len = len(stage_order)

def make_next_stage(world, stage, num):
    base = 'SuperMarioBros-'
    tail = '-v0'

    if num < stage_order_len:
        world = stage_order[num][0]
        stage = stage_order[num][1]
    else:
        if stage >= 4:
            stage = 1

            if world >= 8:
                world = 1
            else:
                world += 1
        else:
            stage += 1
    new_world = base + str(world) + '-' + str(stage) + tail

    return world, stage, new_world

filepath = './DATA/sessions/session0'
infopath= './DATA/video_info/info0.json'

with open(infopath) as json_file:
    video_info =  json.load(json_file)

cap = cv2.VideoCapture('./DATA/videos/video0.avi')

with open(filepath) as json_file:
    data = json.load(json_file)
    #print(data['obs'])


first_world = 'SuperMarioBros-1-1-v0'
env = gym_super_mario_bros.make(first_world)

#env = gym_super_mario_bros.make('SuperMarioBros-v0')
#env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v1') 
#env = gym_super_mario_bros.make('SuperMarioBros-v2')
#env = JoypadSpace(env, SIMPLE_MOVEMENT)
#env = JoypadSpace(env, RIGHT_ONLY)

#clock.set_fps_limit(env.metadata['video.frames_per_second'])

#print('Actionspace: ', env.action_space.n)
#
next_state = env.reset()
start = time.time()

world = 1
stage = 1
stage_num = 0


video_frame_length = 1 / 30
video_start = video_info['start_time']
video_stop = video_info['stop_time']
game_start = data['start_time']
game_stop = data['stop_time']

video_time = video_stop - video_start
game_time = game_stop - game_start


print('Frame: ' + str(video_frame_length))
print('VT:' + str(video_time))
print('GT:' + str(game_time))
print('VS:' + str(video_start))
print('GS:' + str(game_start))

skipped_frames = 0
while video_start < game_start:
    ret, frame = cap.read()
    video_start += video_frame_length
    skipped_frames += 1

print('Skipped: ' + str(skipped_frames))
print('VS:' + str(video_start))
print('GS:' + str(game_start))



states = []

is_first = True
no = 1

steps = 0

for action in data['obs']:
    #clock.tick()   
    
    env.render()
    
    next_state, reward, done, info = env.step(action)
    steps += 1
    
    #Capture 2 game-frames for each video-frame
    """
    cvt_state = cv2.cvtColor(next_state, cv2.COLOR_BGR2RGB)  
    if is_first:
        cv2.imwrite("./DATA/game_frames/state" + str(no) + "-1.png", cvt_state)
        is_first = False
    else:
        cv2.imwrite("./DATA/game_frames/state" + str(no) + "-2.png", cvt_state)
        ret, frame = cap.read()
        cv2.imwrite("./DATA/video_frames/image" + str(no) + ".png", frame)
        is_first = True
        no += 1
    """


    #Capture 1 game-frames for each video-frame by skipping every 2nd frame
    cvt_state = cv2.cvtColor(next_state, cv2.COLOR_BGR2RGB)  
    if is_first:
        #cv2.imwrite("./DATA/testFrames/state" + str(no) + "-1.png", cvt_state)
        is_first = False
    else:
        cv2.imwrite("./DATA/game_frames/state" + str(no) + ".png", cvt_state)
        #cv2.imwrite("./home/henriksv/Documents/RL/mario-data-collection/mario-data-collection/data_processing/game_frames/state" + str(no) + ".png", cvt_state)
        ret, frame = cap.read()
        cv2.imwrite("./DATA/video_frames/image" + str(no) + ".png", frame)
        is_first = True
        no += 1
    
    #states.append(cvt_state)
    #cv2.imwrite("./DATA/testFrames/state" + str(no) + ".png", cvt_state)
    #no += 1
    
    
    if info['flag_get']:
        finish = True

    if done:
        done = False
        end = time.time()
        
        if finish or steps >= 16000:
                    stage_num += 1
                    world, stage, new_world = make_next_stage(world, stage, stage_num)
                    env.close()
                    env = gym_super_mario_bros.make(new_world)
                    finish = False
                    steps = 0

        next_state = env.reset()
