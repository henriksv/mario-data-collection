import matplotlib.pyplot as plt
import csv
import json
from scipy.signal import find_peaks


session_fname = 'session3'
session_path = './DATA/sessions/' + session_fname
e4_path = './DATA/e4/03/BVP.csv'

with open(session_path) as json_file:
    session = json.load(json_file)

session_start = session['start_time']
session_stop = session['stop_time']

bvp = []
with open(e4_path, newline='') as csvfile:
    rd = csv.reader(csvfile)
    for row in rd:
        bvp.append(row[0])

e4_start = float(bvp[0])
samples_pr_sec = bvp[1]

#print(e4_start)
print(samples_pr_sec)
#print(len(bvp))
x = []
y = []
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

#print(j)
#print(k)
print(len(session['obs']))
print(e4_start)
print(session_start)
print(e4_end)
print(session_stop)

for i in range(j,k):
#for i in range(2,len(bvp)):
    if(i % 1 == 0):
        y.append(float(bvp[i]))
        x.append(0 + timestep*(i-j))


max_y = max(y)
min_y = min(y) # Normalize with actual min
#min_y = -max_y # Normalize with reverse of max as min

range_y = max_y - min_y

#print(max_y)
#print(min_y)
#print(range_y)

normalized_y = []

print(max_y)
print(min_y)

for i in range(len(y)):
    if y[i] < min_y:
        normalized_y.append(-1)
    else:
        norm_val = y[i] / max_y
        normalized_y.append(norm_val)

print(max(normalized_y))
print(min(normalized_y))

"""
#plt.plot(x,y, label='BVP')
plt.plot(x,normalized_y, label='BVP (Normalized)')
plt.xlabel('time')
plt.ylabel('BVP')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()
"""

normalized_y_positive = []

for i in range(0, len(normalized_y)):
    if normalized_y[i] >= 0 :
        normalized_y_positive.append(normalized_y[i])
    else:
        normalized_y_positive.append(0)



peaks, _ = find_peaks(normalized_y, distance=40)

print(peaks)

single_peak_values = []
single_peak_values_with_min = []
cur_val = 0.5

start = 0
last_val = 0

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


#print('Len of spv2:', len(single_peak_values2))


fname = session_fname + '_est_bvp_amps.csv'

with open(fname, 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(single_peak_values_with_min)

    #for val in single_peak_values_with_min:
     #   wr.writerow(str(val))




"""
plt.subplot(5, 1, 1)
plt.plot(x, y, label='BVP (Normalized)', linewidth=1)
plt.title('BVP plots')
plt.ylabel('BVP')

x_step = range(0, len(x))

plt.subplot(5, 1, 2)
plt.plot(x_step, normalized_y)#, '.-', linewidth=1)
plt.xlabel('time (s)')
plt.ylabel('BVP(Normalized)')
#plt.plot(peaks, x_step[peaks], "x")

plt.subplot(5, 1, 3)
plt.plot(x_step, normalized_y_positive)#, '.-', linewidth=1)
plt.xlabel('time (s)')
plt.ylabel('BVP(Normalized to positive)')
#plt.plot(peaks, x_step[peaks], "x")

plt.subplot(5, 1, 4)
plt.plot(x_step, single_peak_values, '.-', linewidth=1)
plt.xlabel('steps')
plt.ylabel('Vasoconstriction')
#plt.plot(peaks, x_step[peaks], "x")

plt.subplot(5, 1, 5)
plt.plot(x_step, single_peak_values_with_min, '.-', linewidth=1)
plt.xlabel('steps')
plt.ylabel('Vasoconstriction with minimum')
#plt.plot(peaks, x_step[peaks], "x")


#leg = plt.legend()
plt.show()

#https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/annotation_demo.html

"""