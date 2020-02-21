import matplotlib.pyplot as plt
import csv
import json
from scipy.signal import find_peaks


session_path = './DATA/sessions/session0'

with open(session_path) as json_file:
    session = json.load(json_file)

session_start = session['start_time']
session_stop = session['stop_time']

print(session_start)
print(session_stop)

bvp = []
with open('./DATA/e4/00_1579783503_A00D58/BVP.csv', newline='') as csvfile:
    rd = csv.reader(csvfile)
    for row in rd:
        #print(', '.join(row))
        bvp.append(row[0])

e4_start = float(bvp[0])
samples_pr_sec = bvp[1]

#print(e4_start)
#print(samples_pr_sec)
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
#print(e4_start)
#print(session_start)
#print(e4_end)
#print(session_stop)

for i in range(j,k):
#for i in range(2,len(bvp)):
    if(i % 1 == 0):
        y.append(float(bvp[i]))
        x.append(0 + timestep*(i-j))


max_y = max(y)

#min_y = min(y) # Normalize with actual min

min_y = -max_y # Normalize with reverse of max as min


range_y = max_y - min_y

print(max_y)
print(min_y)
print(range_y)


normalized_y = []

for i in range(len(y)):

    if y[i] < min_y:
        normalized_y.append(-1)
    else:
        norm_val = y[i] / max_y
        normalized_y.append(norm_val)

"""
#plt.plot(x,y, label='BVP')
plt.plot(x,normalized_y, label='BVP (Normalized)')
plt.xlabel('time')
plt.ylabel('BVP')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()
"""





peaks, _ = find_peaks(normalized_y, distance=40)
#print(peaks)

single_peak_values = []
single_peak_values2 = []
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

print('Len of spv:', len(single_peak_values))
print('Len of ny:', len(normalized_y))
        
cur_val = 0.5
start = 0
for i in range(0, len(normalized_y)):
    if i in peaks:
        new_val = normalized_y[i]
        if new_val > 0.1:
            cur_val = new_val
            
        for j in range(start, i+1):
            single_peak_values2.append(cur_val)
        start = i+1
        
    if i == len(normalized_y)-1:
        for j in range(start, i+1):
            single_peak_values2.append(cur_val)


print('Len of spv2:', len(single_peak_values2))
"""
for i in range(0, len(normalized_y)):
    if i in peaks:
        new_val = normalized_y[i]
        if new_val < 0:
            cur_val = 0
        else:
            cur_val = normalized_y[i]
    single_peak_values.append(cur_val)

cur_val = 0.5

for i in range(0, len(normalized_y)):
    if i in peaks:
        new_val = normalized_y[i]
        if new_val > 0.1:
            cur_val = normalized_y[i]
    single_peak_values2.append(cur_val)
"""

normalized_y_positive = []

for i in range(0, len(normalized_y)):
    if normalized_y[i] > 0 :
        normalized_y_positive.append(normalized_y[i])
    else:
        normalized_y_positive.append(0)

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
plt.plot(x_step, single_peak_values2, '.-', linewidth=1)
plt.xlabel('steps')
plt.ylabel('Vasoconstriction')
#plt.plot(peaks, x_step[peaks], "x")


leg = plt.legend()
plt.show()

#https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/annotation_demo.html

"""
i = 0
last_peak = None
last_valley = None
rising = None

skip_start = 0
skip_end = 0

last = len(normalized_y)-1
vasoconstriction = 0

while i < last-1:
    if i == 0: 
        # Case of falling start
        if normalized_y[i] > normalized_y[i+1]:
            while normalized_y[i] > normalized_y[i+1]:
                i+=1
                e4_start += timestep
                skip_start += 1
            last_valley = normalized_y[i]
            rising = True
            

        # Case of rising start
        else:
            while normalized_y[i] <= normalized_y[i+1]:
                i+=1
                e4_start += timestep
                skip_start += 1
            last_peak = normalized_y[i]
            rising = False
    else:
        if rising:
            peak = i
            valley = i

            while normalized_y[peak] < normalized_y[peak+1]:
                if peak >= last-1:
                    print('last')
                    skip_end = last - valley
                    i = last
                    single_peak_values.append(vasoconstriction)
                    break

                peak+=1


            valley_val = normalized_y[valley]
            peak_val = normalized_y[peak]

            vasoconstriction = peak_val - valley_val
            last_peak = peak
            i = peak
            rising = False

            for j in range(valley, peak):
                single_peak_values.append(vasoconstriction)
        else:
            peak = i
            valley = i

            while normalized_y[valley] >= normalized_y[valley+1]:
                if valley >= last-1:
                    print('last')
                    skip_end = last - peak
                    i = last
                    single_peak_values.append(vasoconstriction)
                    break
                valley+=1

            valley_val = normalized_y[valley]
            peak_val = normalized_y[peak]

            vasoconstriction = peak_val - valley_val
            last_valley = valley
            i = valley
            rising = True
            
            for j in range(peak, valley):
                single_peak_values.append(vasoconstriction)

print(skip_start)
print(skip_end)
    
print(len(single_peak_values))

y_cropped = len(normalized_y) - skip_end - skip_start
print(y_cropped)


for i in range(0, 200):
    print(normalized_y[last-i])
    #print(single_peak_values[i])

"""

#x2 = range(0, len(normalized_y))
