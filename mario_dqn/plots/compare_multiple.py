import matplotlib.pyplot as plt
import numpy as np
import gym
import json

def compare(paths, legends, metric, label, test, limit=10000, avg_n=500, leg_pos='high', tail=""):

    out_path = 'CompareMult_'+metric+'_'+str(limit)+'_'+test+'.png'
    if avg_n != 500:    
        out_path = 'CompareMult_avg' + str(avg_n )+'_'+ metric+'_'+str(limit)+'_'+ test + tail + '.png'
    

    data = []
    for p in paths:
        with open(p) as json_file:
            data.append(json.load(json_file))

    vals = []
    for d in data:
        vals.append(d[metric][0:limit])

    for v in vals:
        print(len(v))

    i = 0
    x = [i+1 for i in range(0, limit)]#len(vals))]
    if metric == 'scores':   
        x = [i+1 for i in range(avg_n, limit)]#len(vals))]


    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")

    avgs = []

    for v in vals:
        N = len(v)
        running_avg = np.empty(N)
        for t in range(N):
            running_avg[t] = np.mean(v[max(0, t-avg_n):(t+1)])
        #For score
        if metric == 'scores':
            running_avg = running_avg[avg_n:]
        avgs.append(running_avg)

    for i, avg in enumerate(avgs):
        color = 'C' + str(i+1)
        ax.plot(x, avg, color=color, label=legends[i])


    ax.set_xlabel("Episode", color="C0")
    ax.set_ylabel(label, color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    if leg_pos == 'low':
        ax.legend(loc='lower right')#, frameon=False)
    else:
        ax.legend(loc='upper left')#, frameon=False)


    plt.savefig(out_path)
    plt.close()



# Orange, green, red, purple

"""#m100k_lr0.00001_ed0.99995 - 30k -running
path1 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_m100k_lr0.00001_ed0.99995-2/mario_DDQN_m100k_lr0.00001_ed0.99995-2_episodes.json'
#m100k_lr0.00005_ed0.99995 - 40k
path2 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_m100k_lr0.00005_ed0.99995-2/mario_DDQN_m100k_lr0.00005_ed0.99995-2_episodes.json'
#m100k_lr0.0001_ed0.99995 - 40k
path3 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_m100k_lr0.0001_ed0.99995-2/mario_DDQN_m100k_lr0.0001_ed0.99995-2_episodes.json'
#m100k_lr0.00025_ed0.99995 - 18k
path4 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_m100k_ed0.99995-2/mario_DDQN_m100k_ed0.99995-2_episodes.json'
#m100k_lr0.001_ed0.99995 - 14k
path5 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_m100k_lr0.001_ed0.99995/mario_DDQN_m100k_lr0.001_ed0.99995_episodes.json'
""""""
paths = [path1, path2, path3, path4, path5]
legends = ['Agent 1: LR:0.00001', 'Agent 2: LR:0.00005', 'Agent 3: LR:0.0001', 'Agent 4: LR:0.00025', 'Agent 5: LR:0.001']
label = 'Finish Rate (pr. 500 eps)'
test = 'lr'
compare(paths, legends, 'success', label,test, 10000)
label = 'Finish Rate (pr. 1000 eps)'
compare(paths, legends, 'success', label,test, 10000, 1000)

label = 'Average Score (pr. 500 eps)'
compare(paths, legends, 'scores', label,test, 10000)
label = 'Average Score (pr. 1000 eps)'
compare(paths, legends, 'scores', label,test, 10000, 1000)

paths = [path1, path2, path3]
leg_pos = 'low'
legends = ['Agent 1: LR:0.00001', 'Agent 2: LR:0.00005', 'Agent 3: LR:0.0001', 'Agent 4: LR:0.00025', 'Agent 5: LR:0.001']
label = 'Finish Rate (pr. 500 eps)'
compare(paths, legends, 'success', label,test, 40000, leg_pos)
label = 'Finish Rate (pr. 1000 eps)'
compare(paths, legends, 'success', label,test, 40000, 1000, leg_pos)

label = 'Average Score (pr. 500 eps)'
compare(paths, legends, 'scores', label,test, 40000, leg_pos)
label = 'Average Score (pr. 1000 eps)'
compare(paths, legends, 'scores', label,test, 40000, 1000, leg_pos)"""

#Compare memory
"""
#m50k_lr0.00005
path6 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_m50k_lr0.00005/mario_DDQN_m50k_lr0.00005_episodes.json'
#m250k_lr0.00005
path7 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_m250k_lr0.00005/mario_DDQN_m250k_lr0.00005_episodes.json'
#m500k
path8 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_m500k_lr0.00005/mario_DDQN_m500k_lr0.00005_episodes.json'
#m10k
path9 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_m10k_lr0.00005/mario_DDQN_m10k_lr0.00005_episodes.json'
#m25k
path10 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_m25k_lr0.00005/mario_DDQN_m25k_lr0.00005_episodes.json'

test = 'mem'"""
"""
paths = [path6, path2, path7, path8]
legends = ['Agent 1: Mem:50k', 'Agent 2: Mem:100k', 'Agent 3: Mem:250k', 'Agent 4: Mem:500k']

label = 'Finish Rate (pr. 500 eps)'
compare(paths, legends, 'success', label,test, 17000)
label = 'Finish Rate (pr. 1000 eps)'
compare(paths, legends, 'success', label,test, 17000, 1000)

label = 'Average Score (pr. 500 eps)'
compare(paths, legends, 'scores', label,test, 17000)
label = 'Average Score (pr. 1000 eps)'
compare(paths, legends, 'scores', label,test, 17000, 1000)"""
"""
paths = [path9, path10, path6, path2, path7, path8]
legends = ['Agent 1: Mem:10k', 'Agent 2: Mem:25k', 'Agent 3: Mem:50k', 'Agent 4: Mem:100k', 'Agent 5: Mem:250k', 'Agent 6: Mem:500k']

n = 17000
pos = 'low'
label = 'Finish Rate (pr. 500 eps)'
compare(paths, legends, 'success', label,test, n, leg_pos=pos)
label = 'Finish Rate (pr. 1000 eps)'
compare(paths, legends, 'success', label,test, n, 1000, leg_pos=pos)

label = 'Average Score (pr. 500 eps)'
compare(paths, legends, 'scores', label,test, n, leg_pos=pos)
label = 'Average Score (pr. 1000 eps)'
compare(paths, legends, 'scores', label,test, n, 1000, leg_pos=pos)
"""

"""test = 'qlty'

path11 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_up60_m25k_lr0.00005/mario_DDQN_up60_m25k_lr0.00005_episodes.json'
path12 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_up120_m25k_lr0.00005/mario_DDQN_up120_m25k_lr0.00005_episodes.json'
""""""
paths = [path11, path10, path12]
legends = ['Agent 1: 60x60', 'Agent 2: 84x84', 'Agent 3: 120x120']

n = 15000
pos = 'low'
label = 'Finish Rate (pr. 500 eps)'
compare(paths, legends, 'success', label,test, n, leg_pos=pos)
label = 'Finish Rate (pr. 1000 eps)'
compare(paths, legends, 'success', label,test, n, 1000, leg_pos=pos)

label = 'Average Score (pr. 500 eps)'
compare(paths, legends, 'scores', label,test, n, leg_pos=pos)
label = 'Average Score (pr. 1000 eps)'
compare(paths, legends, 'scores', label,test, n, 1000, leg_pos=pos)"""

test = 'emo'
pathX = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_m10k_lr0.00005/mario_DDQN_m10k_lr0.00005_episodes.json'

path0 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_EDDQN_p0_0.5/mario_EDDQN_p0_0.5_episodes.json'
path1 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_EDDQN_p1_0.5_v4/mario_EDDQN_p1_0.5_v4_episodes.json'
path2 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_EDDQN_p2_0.5/mario_EDDQN_p2_0.5_episodes.json'
path3 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_EDDQN_p3_0.5_v4/mario_EDDQN_p3_0.5_v4_episodes.json'
path4 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_EDDQN_p4_0.5/mario_EDDQN_p4_0.5_episodes.json'
path5 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_EDDQN_p5_0.5_v3/mario_EDDQN_p5_0.5_v3_episodes.json'
path6 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_EDDQN_p6_0.5/mario_EDDQN_p6_0.5_episodes.json'
path7 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_EDDQN_p7_0.5_v4/mario_EDDQN_p7_0.5_v4_episodes.json'
path8 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_EDDQN_p8_0.5/mario_EDDQN_p8_0.5_episodes.json'
path9 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_EDDQN_p9_0.5/mario_EDDQN_p9_0.5_episodes.json'
"""
# Best
paths = [
    pathX, 
    path0, #17000
    path1, #10000
    #path2, #10000
    #path3, #10000
    #path4, #10000
    #path5, #17000
    path6, #17000
    path7, 
    #path8, #11000
    path9  #10000
    ]

n = 10000
legends = [
    'Agent X: vanilla', 
    'Agent 0: p0 ',
    'Agent 1: p1 ', 
    'Agent 6: p6 ', 
    'Agent 7: p7 ', 
    'Agent 9: p9 '
    ]

tail = "-BEST"
pos = 'low'
label = 'Finish Rate (pr. 1000 eps)'
compare(paths, legends, 'success', label,test, n, 1000, leg_pos=pos, tail=tail)

label = 'Average Score (pr. 1000 eps)'
compare(paths, legends, 'scores', label,test, n, 1000, leg_pos=pos, tail=tail)

#Worst
paths = [
    pathX, 
    path2, #10000
    path3, #10000
    path4, #10000
    path5, #17000
    path8, #11000
    ]

n = 10000
legends = [
    'Agent X: vanilla', 
    'Agent 2: p2 ', 
    'Agent 3: p3 ', 
    'Agent 4: p4 ', 
    'Agent 5: p5 ', 
    'Agent 8: p8 ', 
    ]

tail = "-WORST"

pos = 'low'
label = 'Finish Rate (pr. 1000 eps)'
compare(paths, legends, 'success', label,test, n, 1000, leg_pos=pos, tail=tail)

label = 'Average Score (pr. 1000 eps)'
compare(paths, legends, 'scores', label,test, n, 1000, leg_pos=pos, tail=tail)


"""
path01 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_EDDQN_p0_0.25/mario_EDDQN_p0_0.25_episodes.json'
path02 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_EDDQN_p0_0.75_v3/mario_EDDQN_p0_0.75_v3_episodes.json'

path61 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_EDDQN_p6_0.25/mario_EDDQN_p6_0.25_episodes.json'
path62 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_EDDQN_p6_0.75_v4/mario_EDDQN_p6_0.75_v4_episodes.json'


paths = [
    pathX, 
    path01,
    path0,  
    path02,
    ]


test = 'weight_p0'
n = 10000
legends = [
    'Agent X: vanilla', 
    'Agent 0: p0 - weight=0.25 ',
    'Agent 1: p0 - weight=0.5 ', 
    'Agent 2: p0 - weight=0.75 '
    ]

pos = 'high'
label = 'Finish Rate (pr. 1000 eps)'
compare(paths, legends, 'success', label,test, n, 1000, leg_pos=pos)

pos = 'low'
label = 'Average Score (pr. 1000 eps)'
compare(paths, legends, 'scores', label,test, n, 1000, leg_pos=pos)

paths = [
    pathX, 
    path61, 
    path6,  
    path62
    ]


test = 'weight_p6'
n = 10000
legends = [
    'Agent X: vanilla', 
    'Agent 3: p6 - weight=0.25 ', 
    'Agent 4: p6 - weight=0.5 ', 
    'Agent 5: p6 - weight=0.75 '
    ]

pos = 'high'
label = 'Finish Rate (pr. 1000 eps)'
compare(paths, legends, 'success', label,test, n, 1000, leg_pos=pos)
pos = 'low'
label = 'Average Score (pr. 1000 eps)'
compare(paths, legends, 'scores', label,test, n, 1000, leg_pos=pos)
"""
   episode_history = {
                'epsilon' : epsilon_history,
                'scores' : scores,
                'success' : success_history,
                'position' : pos_history,
                'time' : time_history,
                'progression' : progression_history
            }"""