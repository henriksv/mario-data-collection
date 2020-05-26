import matplotlib.pyplot as plt
import numpy as np
import gym
import json

# Orange, green, red, purple

#m100k_lr0.00001_ed0.99995 - 30k -running
path1 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_m100k_lr0.00001_ed0.99995-2/mario_DDQN_m100k_lr0.00001_ed0.99995-2_episodes.json'

#m100k_lr0.00005_ed0.99995 - 40k
path2 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_m100k_lr0.00005_ed0.99995-2/mario_DDQN_m100k_lr0.00005_ed0.99995-2_episodes.json'

#m100k_lr0.0001_ed0.99995 - 40k
path3 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_m100k_lr0.0001_ed0.99995-2/mario_DDQN_m100k_lr0.0001_ed0.99995-2_episodes.json'

#m100k_lr0.00025_ed0.99995 - 18k
path4 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_m100k_ed0.99995-2/mario_DDQN_m100k_ed0.99995-2_episodes.json'

#m100k_lr0.001_ed0.99995 - 14k
path5 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_m100k_lr0.001_ed0.99995/mario_DDQN_m100k_lr0.001_ed0.99995_episodes.json'

paths = [path1, path2, path3, path4, path5]
#paths = [path1, path2, path3]

legends = ['LR:0.00001', 'LR:0.00005', 'LR:0.0001', 'LR:0.00025', 'LR:0.001']
#legends = ['LR:0.00001', 'LR:0.00005', 'LR:0.0001']



#path1 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_m100k_ed0.99995-2/mario_DDQN_m100k_ed0.99995-2_episodes.json'
#path2 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_m100k_ed0.99995-1/mario_DDQN_m100k_ed0.99995-1_episodes.json'
#path2 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_testingDQN_m50k_ed0.99995/mario_DDQN_testingDQN_m50k_ed0.99995_episodes.json'

#paths = [path1, path2]#, path5, path4, path5]

limit = 10000
avg_n=500
metric = 'scores'
label = 'Avg Score (pr. 500 eps)'
out_path = 'CompareMult_'+metric+'_10k_lr.png'

data = []
for p in paths:
    with open(p) as json_file:
        data.append(json.load(json_file))

vals = []
for d in data:
    vals.append(d[metric][0:limit])

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

ax. legend(loc='upper left')#, frameon=False)

plt.savefig(out_path)
plt.close()

"""
   episode_history = {
                'epsilon' : epsilon_history,
                'scores' : scores,
                'success' : success_history,
                'position' : pos_history,
                'time' : time_history,
                'progression' : progression_history
            }"""