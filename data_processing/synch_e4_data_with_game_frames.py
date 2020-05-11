import json
import csv
import os


def synch_e4_by_pruning(e4_type, session_num):
    session_number = str(session_num)
    session_path = './DATA/sessions/session' + session_number
    with open(session_path) as json_file:
        session = json.load(json_file)

    gap_info_path = './DATA/gap_info/gap_info'+session_number
    with open(gap_info_path) as json_file:
        gap_info = json.load(json_file)

    e4_path = './DATA/e4/'+session_number+'/'+e4_type+'.csv'
    e4 = []
    with open(e4_path, newline='') as csvfile:
        rd = csv.reader(csvfile)
        for row in rd:
            e4.append(row)
  
    e4_pr_sec = float(e4[1][0])
    e4_start = float(e4[0][0])

    session_start = session['start_time']
    session_stop = session['stop_time']


    n_frames = len(session['obs'])
    missing = gap_info['missing']
    gap_indices = gap_info['indices']
    n_gaps = len(gap_indices)
    avg_gap = int(missing/n_gaps)
    extra = missing % n_gaps
    frame_pr_sek = 60
    
    timestep = 1 / e4_pr_sec
    j = 2

    while e4_start < session_start-timestep/2:
        e4_start += timestep
        j +=1

    k = j
    e4_end = e4_start

    while e4_end < session_stop-timestep/2:
        e4_end += timestep
        k +=1


    synched_e4 = []
    for i in range(j,k):
        synched_e4.append(e4[i])
        
    if e4_type == "BVP" or e4_type == "ACC":
        pruned_e4 = []
        for i in range(len(synched_e4)):
            if i % 16 != 0:
                pruned_e4.append(synched_e4[i])
        synched_e4 = pruned_e4
        e4_pr_sec = 60
        if e4_type == "ACC":
            e4_pr_sec = 30

    frames_pr_e4 = int(frame_pr_sek / e4_pr_sec)
    transformed_e4 = []
    times = []

    for i in range(j,k):
        for l in range(frames_pr_e4):
            transformed_e4.append(e4[i])
            times.append(0 + (1/60)*(i-j))
        
    #print(len(transformed_e4))

    matching_e4 = []
    n = 0
    for i in range(n_frames):
        matching_e4.append(transformed_e4[n])

        if i in gap_indices:
            n += avg_gap
            if extra > 0:
                n += 1
                extra -= 1

        i += 1
        n += 1

    #print(n_frames)
    #print(len(matching_e4))

    out_file_path = './DATA/e4_synced/'+session_number+'/'+ e4_type+'.csv'
    with open(out_file_path, "w") as f:
        for value in matching_e4:
            f.write("%s\n" % value)

def synch_e4_with_session_time(e4_type, session_num):
    session_number = str(session_num)
    session_path = './DATA/sessions/session' + session_number
    with open(session_path) as json_file:
        session = json.load(json_file)

    gap_info_path = './DATA/gap_info/gap_info'+session_number
    with open(gap_info_path) as json_file:
        gap_info = json.load(json_file)

    e4_path = './DATA/e4/'+session_number+'/'+e4_type+'.csv'
    e4 = []
    with open(e4_path, newline='') as csvfile:
        rd = csv.reader(csvfile)
        for row in rd:
            e4.append(row)

    e4_pr_sec = float(e4[1][0])
    e4_start = float(e4[0][0])

    session_start = session['start_time']
    session_stop = session['stop_time']
    
    timestep = 1 / e4_pr_sec
    j = 2
    while e4_start < session_start-timestep/2:
        e4_start += timestep
        j +=1

    k = j
    e4_end = e4_start

    while e4_end < session_stop-timestep/2:
        e4_end += timestep
        k +=1


    synched_e4 = []
    for i in range(j,k):
        synched_e4.append(e4[i])

    #Save the synced e4 data
    dir_path = './DATA/e4_synchronized/'+session_number + '/'
    out_file_path = dir_path + e4_type+'.csv'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    with open(out_file_path, "w") as f:
        for value in synched_e4:
            f.write("%s\n" % value)
    
    return synched_e4
        

if __name__ == "__main__":
    #Example of sync with padded game frames for all supported datatypes for one session
    e4_types = ['EDA', 'TEMP', 'HR', 'ACC', 'BVP']

    for metric in e4_types:
        vals = synch_e4_with_session_time(metric, 6)
        print(metric + ' values: ' + str(len(vals)))
        