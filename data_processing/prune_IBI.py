import matplotlib.pyplot as plt
import csv
import json
from scipy.signal import find_peaks

def prune_IBI():
    session_fname = 'session6'
    session_path = './DATA/sessions/' + session_fname
    ibi_path = './DATA/e4/6/IBI.csv'

    with open(session_path) as json_file:
        session = json.load(json_file)

    session_start = session['start_time']
    session_stop = session['stop_time']
    timestep = 1 / 60

    ibi = []
    with open(ibi_path, newline='') as csvfile:
        rd = csv.reader(csvfile)
        for row in rd:
            #print(row)
            ibi.append(row)

    ibi_start = float(ibi[0][0])
    ibi_time = ibi_start
    pruned_ibi = []

    i = 0
    while ibi_time < session_start:
        i +=1
        ibi_time = ibi_start + float(ibi[i][0])

    skipped_time = float(ibi[i][0]) - float(ibi[i][1])


    pruned_ibi.append([ibi_time - float(ibi[i][1]), 'IBI'])

    while ibi_time < session_stop and i < len(ibi):
        pruned_ibi.append([float(ibi[i][0])-skipped_time, ibi[i][1]])
        i += 1

    print(pruned_ibi)

    fname = './DATA/' + session_fname + '_pruned_IBI.csv'

    with open(fname, 'w') as myfile:
        print('writing to: ' + fname)
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(pruned_ibi)

prune_IBI()