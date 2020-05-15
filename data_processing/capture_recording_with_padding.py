
import gym_super_mario_bros
import numpy as np
import gym

import json
from pyglet import clock
import time
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from stages import stage_order, make_next_stage


def capture_recording_with_padding(session_number, downscale_game=False, downscale_game_dim=84, grayscale_game=False,
                                                downscale_video=False, downscale_video_dim=120, grayscale_video=False):

    session_number = str(session_number)
    st_width =  256
    st_height = 240

    filepath = './DATA/sessions/session'+session_number
    infopath= './DATA/video_info/info'+session_number+'.json'

    with open(infopath) as json_file:
        video_info =  json.load(json_file)

    cap = cv2.VideoCapture('./DATA/videos/video'+session_number+'.avi')

    with open(filepath) as json_file:
        data = json.load(json_file)
        #print(data['obs'])

    print('...Capturing recording with padding...')
    first_world = 'SuperMarioBros-1-1-v0'
    env = gym_super_mario_bros.make(first_world)
    next_state = env.reset()
    #start = time.time()

    world = 1
    stage = 1
    stage_num = 0

    video_frame_length = 1 / 30
    video_start = video_info['start_time']
    game_start = data['start_time']
    video_stop = video_info['stop_time']
    game_stop = data['stop_time']

    """
    video_time = video_stop - video_start
    game_time = game_stop - game_start
    print('Frame: ' + str(video_frame_length))
    print('VT:' + str(video_time))
    print('GT:' + str(game_time))
    print('VS:' + str(video_start))
    print('GS:' + str(game_start))"""

    skipped_video_frames = 0
    extra_game_frames = 0
    while video_start + (video_frame_length/2) < game_start:
        ret, frame = cap.read()
        video_start += video_frame_length
        skipped_video_frames += 1

        if skipped_video_frames > 30:
            extra_game_frames +=2

    print('Skipped video frames: ' + str(skipped_video_frames))
    print('Extra game frames: ' + str(extra_game_frames))
    print('Video Start:' + str(video_start))
    print('Game Start:' + str(game_start))
    
    is_first = True
    no = 0
    finish = False
    steps = 0
    total_steps = 0
    gap_indices = []
    n_actions = len(data['obs'])


    print('...extracting game frames...')
    frame_dir_path = "./DATA/game_frames/"+session_number
    if not os.path.exists(frame_dir_path):
        os.mkdir(frame_dir_path)

    for action in data['obs']:
        if total_steps == n_actions-extra_game_frames:
            break
        #env.render()
        next_state, reward, done, info = env.step(action)
        steps += 1
        total_steps += 1
        
        cvt_state = cv2.cvtColor(next_state, cv2.COLOR_BGR2RGB)
        if grayscale_game:
            cvt_state = cv2.cvtColor(cvt_state, cv2.COLOR_RGB2GRAY)
        if downscale_game:
            cvt_state = cv2.resize(cvt_state, (downscale_game_dim, downscale_game_dim), interpolation=cv2.INTER_AREA)
        
        #Capture 1 game-frames for each video-frame by skipping every 2nd frame
        if is_first:
            #cv2.imwrite("./DATA/testFrames/state" + str(no) + "-1.png", cvt_state)
            is_first = False
        else:
            cv2.imwrite("./DATA/game_frames/"+session_number+"/state" + str(no) + ".png", cvt_state)
            is_first = True
            no += 1
        
        if info['flag_get']:
            finish = True

        if done:
            done = False
     #       end = time.time()
            
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
    missing = 126000 - n_actions
    video_frames_to_skip = missing/2
    avg_gap_len = int(video_frames_to_skip / n_gaps)
    extra = video_frames_to_skip % n_gaps
    
    
    #print('..gap info...')
    #print('N gaps:'+str(n_gaps))
    #print('Frames to skip: '+ str(video_frames_to_skip))
    #print('gap len: ' + str(avg_gap_len))
    #print('extra: ' + str(extra))
    

    def write_video_frame(index, dir_path):
        ret, frame = cap.read()
        if grayscale_video:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if downscale_video:
            frame = cv2.resize(frame, (downscale_video_dim, downscale_video_dim), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dir_path+"/image" + str(index) + ".png", frame)

    def write_new_game_frame(index):
        if downscale_game:
            img = np.zeros([downscale_game_dim,downscale_game_dim,1],dtype=np.uint8)
        else:
            img = np.zeros([st_height, st_width,1],dtype=np.uint8)
        img.fill(0) # or img[:] = 255
        cv2.imwrite("./DATA/game_frames/"+session_number+"/frame" + str(index) + ".png", img)


    k = 0
    no = 0
    skips = 0

    first = True
    print('...extracting video and padding game frames...')
    image_path = "./DATA/video_frames/"
    image_dir_path = image_path+session_number
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    if not os.path.exists(image_dir_path):
        os.mkdir(image_dir_path)
    for i in range(n_actions-extra_game_frames):    
        if first:
            first = False
        else:
            first = True
            write_video_frame(k, image_dir_path)
            frame_path = "./DATA/game_frames/"+session_number+"/state" + str(no) + ".png"
            new_frame_path = "./DATA/game_frames/"+session_number+"/frame" + str(k) + ".png"
            os.rename(frame_path,new_frame_path)
            no += 1
            k += 1

        if i in gap_indices:
            skips += 1
            for j in range(int(avg_gap_len)):            
                write_video_frame(k, image_dir_path)
                write_new_game_frame(k)
                k+=1

            if extra > 0:
                write_new_game_frame(k)
                write_video_frame(k, image_dir_path)
                extra -= 1
                k+=1
        
    print('...saving gap info...')
    gap_info = {}
    gap_info['indices'] = gap_indices
    gap_info['missing'] = missing
    
    gap_dir_path = "./DATA/gap_info/"
    if not os.path.exists(gap_dir_path):
        os.mkdir(gap_dir_path)

    gap_path = gap_dir_path + 'gap_info'+session_number 
    with open(gap_path, 'w') as outfile:
        json.dump(gap_info, outfile)
    print('...data capture complete...')

    
if __name__ == "__main__":
    #Example use with game and video downscaling and grayscaling using default dimensions
    if len(sys.argv) > 1:
        for i in range(1, len(sys.argv)):    
            session_n = sys.argv[i]
            if session_n not in ['0','1','2','3','4','5','6','7','8','9']:
                print('Invalid session! ' + session_n + ' is not a valid session number.')
            else:
                print('Capturing data from session ' + session_n)
                capture_recording_with_padding(session_n, downscale_game=True, downscale_video=True, grayscale_game=True, grayscale_video=True)
    else:
        print('No session specified. Provide session number(s) to extract.')