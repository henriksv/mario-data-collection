import matplotlib.pyplot as plt
import numpy as np
import gym
import json

def compare_qlt(metric,limit=30000, avg_n=1000, leg_pos='low'):
    #paths, legends, , label, test, , , 

    # 52980: mario_DDQN_up60_m25k_lr0.00005_RERUN
    # 52981: mario_DDQN_m25k_lr0.00005_2
    # 52709: mario_DDQN_up120_m25k_lr0.00005
    
    #placeholder!
    path1 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_up60_m25k_lr0.00005_RERUN/mario_DDQN_up60_m25k_lr0.00005_RERUN_episodes.json'

    path2 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_m25k_lr0.00005_2/mario_DDQN_m25k_lr0.00005_2_episodes.json'
    path3 = '/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_up120_m25k_lr0.00005/mario_DDQN_up120_m25k_lr0.00005_episodes.json'

    paths = [path1, path2, path3]
    legends = ['Agent 1: 60x60', 'Agent 2: 84x84', 'Agent 3: 120x120']
    out_path = 'CompareQlt_'+metric+'.png'

    data = []
    for p in paths:
        with open(p) as json_file:
            data.append(json.load(json_file))

    print(len(data[0][metric]))

    vals = []
    for d in data:
        #vals.append(d[metric][0:limit])
        vals.append(d[metric][0:int(len(d[metric])/2)])



    i = 0
    x_lens = []
    for v in vals:
        x = [i+1 for i in range(0, len(v))]#len(vals))]
        if metric == 'scores':   
            x = [i+1 for i in range(avg_n, len(v))]#len(vals))]"""
        x_lens.append(x)


   

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

    #fig=plt.figure()
    #ax = fig.add_subplot(111, label=str('1'))
    #ax2 = fig.add_subplot(111, label=str('2'))
    #ax3 = fig.add_subplot(111, label=str('3'))
    
    plt.plot(x_lens[0], avgs[0], color='red', label=legends[0])
    plt.plot(x_lens[1], avgs[1], color='blue', label=legends[1])
    plt.plot(x_lens[2], avgs[2], color='orange', label=legends[2])

    if leg_pos == 'low':
        plt.legend(loc='lower right')#, frameon=False)
    else:
        plt.legend(loc='upper left')#, frameon=False)

    if metric == 'success':
        plt.ylabel('Average Finish Rate (pr 1000 eps)')
    else:
        plt.ylabel('Average Score (pr 1000 eps)')
    plt.xlabel('Episodes')

    plt.savefig(out_path)
    plt.close()


met = 'success'
compare_qlt(met)
compare_qlt('scores')

"""
    
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
    compare(paths, legends, 'scores', label,test, 40000, 1000, leg_pos)

    out_path = 'CompareMult_'+metric+'_'+str(limit)+'_'+test+'.png'
    if avg_n != 500:    
        out_path = 'CompareMult_avg' + str(avg_n )+'_'+ metric+'_'+str(limit)+'_'+test+'.png'
    




    ax.set_xlabel("Episode", color="C0")
    ax.set_ylabel(label, color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    if leg_pos == 'low':
        ax.legend(loc='lower right')#, frameon=False)
    else:
        ax.legend(loc='upper left')#, frameon=False)


   




    Geb_b30 = [11, 10, 12, 14, 16, 19, 17, 14, 18, 17]
    years_b30 = range(2008,2018)
    Geb_a30 = [12, 10, 13, 14, 12, 13, 18, 16]
    years_a30 = range(2010,2018)

    fig, ax = plt.subplots()
    ax.plot(years_b30, Geb_b30, label='Prices 2008-2018', color='blue')
    ax.plot(years_a30, Geb_a30, label='Prices 2010-2018', color = 'red')
    legend = ax.legend(loc='center right', fontsize='x-large')
    plt.xlabel('years')
    plt.ylabel('prices')
    plt.title('Comparison of the different prices')
    plt.show()
"""

# Orange, green, red, purple



"""
   episode_history = {
                'epsilon' : epsilon_history,
                'scores' : scores,
                'success' : success_history,
                'position' : pos_history,
                'time' : time_history,
                'progression' : progression_history
            }"""