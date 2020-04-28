import csv
import json
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def normalize_bvp(x, y):
    max_y = max(y)
    min_y = min(y) # Normalize with actual min
    #min_y = -max_y # Normalize with reverse of max as min

    #Normalize positive and negative values relative to 0    
    normalized_y = []
    for i in range(len(y)):
        if y[i] < 0:
            norm_val = y[i] / (min_y*-1)
            normalized_y.append(norm_val)
        else:
            norm_val = y[i] / max_y
            normalized_y.append(norm_val)

    #Extract only positive
    normalized_y_positive = []
    for i in range(0, len(normalized_y)):
        if normalized_y[i] >= 0 :
            normalized_y_positive.append(normalized_y[i])
        else:
            normalized_y_positive.append(0)

    #Locate peaks
    peaks, _ = find_peaks(normalized_y, distance=40)
    #print(peaks)

    single_peak_values = []
    single_peak_values_with_min = []
    cur_val = 0.5
    start = 0

    for i in range(0, len(normalized_y)):
        if i in peaks:
            for j in range(start, i+1):
                cur_val  = normalized_y[i]
                single_peak_values.append(normalized_y[i])     
            start = i+1

        if i == len(normalized_y)-1:
            for j in range(start, i+1):
                single_peak_values.append(cur_val)

    #print('Len of spv:', len(single_peak_values))
    #print('Len of ny:', len(normalized_y))
            
    cur_val = 0.1
    start = 0
    for i in range(0, len(normalized_y)):
        if i in peaks:
            new_val = normalized_y[i]
            if new_val >= 0.1:
                cur_val = new_val
                
            for j in range(start, i+1):
                single_peak_values_with_min.append(cur_val)
            start = i+1
            
        if i == len(normalized_y)-1:
            for j in range(start, i+1):
                single_peak_values_with_min.append(cur_val)
    """
    fname = session_fname + '_est_bvp_amps.csv'

    with open(fname, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(single_peak_values_with_min)
    """
    return single_peak_values_with_min

def synch_bvp(session_number, halve=False, normalize=False):
    #session_number = '6'
    #bvp_path = './session'+session_number+'_est_bvp_amps.csv'
    session_number = str(session_number)
    session_path = './DATA/sessions/session' + session_number
    with open(session_path) as json_file:
        session = json.load(json_file)

    gap_info_path = './DATA/gap_info/gap_info'+session_number
    with open(gap_info_path) as json_file:
        gap_info = json.load(json_file)

    bvp_path = './DATA/e4/'+session_number+'/BVP.csv'
    bvp = []
    """
    with open(bvp_path, newline='') as csvfile:
        rd = csv.reader(csvfile)
        for row in rd:
            for val in row:
                bvp.append(val)
                #print(row)

    #for i in range(10): print(bvp[i])
    """
    with open(bvp_path, newline='') as csvfile:
        rd = csv.reader(csvfile)
        for row in rd:
            bvp.append(row[0])

    e4_start = float(bvp[0])
    samples_pr_sec = bvp[1]

    session_start = session['start_time']
    session_stop = session['stop_time']

    #print(e4_start)
    #print(samples_pr_sec)
    #print(len(bvp))

    timestep = 1 / 64
    j = 2

    while e4_start < session_start-timestep/2:
        e4_start += timestep
        j +=1

    k = j
    e4_end = e4_start

    while e4_end < session_stop-timestep/2:
        e4_end += timestep
        k +=1

    """
    print(len(session['obs']))
    print(str(k-j))
    print(e4_start)
    print(session_start)
    print(e4_end)
    print(session_stop)
    """
    synched_bvp = []
    times = []

    for i in range(j,k):
    #for i in range(2,len(bvp)):
        #if(i % 1 == 0):
        synched_bvp.append(float(bvp[i]))
        times.append(0 + timestep*(i-j))

    if normalize:
        print("Normalizing values")
        synched_bvp = normalize_bvp(times, synched_bvp)
        

    frames = len(session['obs'])
    #print(len(synched_bvp))

    # Prune bvp to match 60 fps
    pruned_bvp = []
    for i in range(len(synched_bvp)):
        if i % 16 != 0:
            pruned_bvp.append(synched_bvp[i])
    #print(len(pruned_bvp))

    #Skipping gaps to deal with issue of inconsistent number of observations/frames in session
    missing = gap_info['missing']
    gap_indices = gap_info['indices']
    n_gaps = len(gap_indices)

    avg_gap_len = int(missing / n_gaps)
    extra = missing % n_gaps

    print('Matching bvp datapoints with game frames')
    i = 0
    j = 0

    matching_bvp = []
    for i in range(frames):
        matching_bvp.append(pruned_bvp[j])
        if i in gap_indices:      
            j += int(avg_gap_len)
            if extra > 0:
                j += 1
                extra -= 1
        i += 1
        j += 1

    #print(len(matching_bvp))

    # Remove every 2. datapoint to match current number of frames
    halved_bvp = []
    if halve:
        for i in range(len(matching_bvp)):
            if i % 2 != 0:
                halved_bvp.append(matching_bvp[i])

        #print(len(halved_bvp))
        matching_bvp = halved_bvp

    return matching_bvp

bv1 = synch_bvp(6, halve=True)
bv2 = synch_bvp(6, halve=True, normalize=True)
