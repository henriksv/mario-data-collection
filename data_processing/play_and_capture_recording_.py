
import gym_super_mario_bros
import numpy as np
import gym

import json
from pyglet import clock
import time

import numpy as np
import cv2
import matplotlib.pyplot as plt

from stages import stage_order, make_next_stage


def downscale(frame, width, height):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame

stage_order_len = len(stage_order)
session_number = "6"
downscaled_h = 84

filepath = './DATA/sessions/session'+session_number
infopath= './DATA/video_info/info'+session_number+'.json'

with open(infopath) as json_file:
    video_info =  json.load(json_file)

cap = cv2.VideoCapture('./DATA/videos/video'+session_number+'.avi')

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
no = 0
finish = False

steps = 0

total_steps = 0
gap_indices = []

for action in data['obs']:
    #clock.tick()   
    
    env.render()
    
    next_state, reward, done, info = env.step(action)
    steps += 1
    total_steps += 1
    
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
    cvt_state = downscale(cvt_state, downscaled_h, downscaled_h)
    if is_first:
        #cv2.imwrite("./DATA/testFrames/state" + str(no) + "-1.png", cvt_state)
        is_first = False
    else:
        cv2.imwrite("./DATA/game_frames/"+session_number+"/state" + str(no) + ".png", cvt_state)
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
            gap_indices.append(total_steps)

        next_state = env.reset()
        
       
#Extract video
n_gaps = len(gap_indices)

n_actions = len(data['obs'])
missing = 126000 - n_actions
video_frames_to_skip = missing/2
avg_gap_len = int(video_frames_to_skip / n_gaps)
extra = video_frames_to_skip % n_gaps

"""
print(n_gaps)
print(video_frames_to_skip)
print(avg_gap_len)
print(extra)
"""
i = 0
skips = 0

first = True
print('Extracting video')
for i in range(n_actions):    
    if first:
        first = False
        i += 1
    else:
        first = True
        ret, frame = cap.read()
        frame = downscale(frame, 120, 120)
        cv2.imwrite("./DATA/video_frames/"+session_number+"/image" + str(i) + ".png", frame)
        i += 1
    if i in gap_indices:
        skips += 1
        for j in range(int(avg_gap_len)):
            ret, frame = cap.read()
        if extra > 0:
            ret, frame = cap.read()
            extra -= 1
    i += 1

print('Saving gap_info')
gap_info = {}
gap_info['indices'] = gap_indices
gap_info['missing'] = missing

gap_path = "./DATA/gap_info/gap_info"+session_number
print('Saving gaps to file')
with open(gap_path, 'w') as outfile:
    json.dump(gap_info, outfile)
