import matplotlib.pyplot as plt
import csv
import json
from scipy.signal import find_peaks


session_fname = 'session0'
session_path = './DATA/sessions/' + session_fname

e4_path = './DATA/e4/1580304574_A00D58/IBI.csv'

with open(session_path) as json_file:
    session = json.load(json_file)

session_start = session['start_time']
session_stop = session['stop_time']
timestep = 1 / 60

ibi = []
with open(e4_path, newline='') as csvfile:
    rd = csv.reader(csvfile)
    for row in rd:
        #print(row)
        ibi.append(row)

ibi_start = float(ibi[0][0])
#base_time = round(ibi_start - float(ibi[1][0]))

base_time = ibi_start
ibi_time = ibi_start



print(session_start)
print(ibi_start)
print(base_time)



i = 0
while ibi_time < session_start:
    i +=1
    ibi_time = ibi_start + float(ibi[i][0])
    
print(ibi_time)
print(len(ibi))
print(i)

print(len(session['obs']))

obs_n = len(session['obs'])

good_beats = []

#for j in range(obs_n):

j = 0
while(j < obs_n):
    game_time = session_start + j*timestep

    if i >= len(ibi):
        good_beats.append(False)
        j+=1
    else:

        covered_time_start = ibi_time - float(ibi[i][1])

        if game_time < covered_time_start or i >= len(ibi):
            good_beats.append(False)
            j+=1
        elif game_time < ibi_time:
            good_beats.append(True)
            j+=1
        else:
            i += 1  
            if(i < len(ibi)):
                ibi_time = base_time + float(ibi[i][0])

print(len(good_beats))

good = 0
bad = 0

for k in range(len(good_beats)):
    if good_beats[k]:
        good += 1
    else:
        bad += 1

print('Good: ', good)
print('Bad: ', bad)
