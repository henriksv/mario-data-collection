import matplotlib.pyplot as plt
import numpy as np
import gym
import json

def plotLearning(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)
    plt.close()

def plot_finish_rate(x, scores, filename, avg_n=100):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    #ax2=fig.add_subplot(111, label="2", frame_on=False)

   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-avg_n):(t+1)])

    ax.plot(x, running_avg, color="C0")
    #ax.scatter(x, running_avg, color="C0", s=2)
    ax.set_xlabel("Episode", color="C0")
    ax.set_ylabel("Finish rate", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")
    

    """ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colo
    """
    plt.savefig(filename + '_FinishRate.png')
    plt.close()

def plot_finish_rate_compare(x, scores, scores2, filename, avg_n=100):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    #ax2=fig.add_subplot(111, label="2")

   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-avg_n):(t+1)])

    N2 = len(scores2)
    running_avg2 = np.empty(N2)
    for t in range(N2):
	    running_avg2[t] = np.mean(scores2[max(0, t-avg_n):(t+1)])

    ax.plot(x, running_avg, color="C0")
    #ax.scatter(x, running_avg, color="C0", s=2)
    ax.set_xlabel("Episode", color="C0")
    ax.set_ylabel("Finish rate", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    ax.plot(x, running_avg2, color="C1")
    #ax.scatter(x, running_avg, color="C0", s=2)
    #ax.set_xlabel("Episode", color="C0")
    #ax.set_ylabel("Finish rate", color="C0")
    #ax2.tick_params(axis='x', colors="C1")
    #ax2.tick_params(axis='y', colors="C1")
    

    """ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colo
    """
    plt.savefig(filename + '_CompareFinishRate.png')
    plt.close()

def plot_scores(x, scores, filename, avg_n=100):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
      
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-avg_n):(t+1)])

    ax.plot(x, running_avg, color="C0")
    #ax.scatter(x, running_avg, color="C0", s=2)
    ax.set_xlabel("Episode", color="C0")
    ax.set_ylabel("Scores", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")
    
    plt.savefig(filename + '_Scores.png')
    plt.close()

def plot_progression(x, scores, filename, avg_n=100):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
      
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-avg_n):(t+1)])

    ax.plot(x, running_avg, color="C0")
    #ax.scatter(x, running_avg, color="C0", s=2)
    ax.set_xlabel("Episode", color="C0")
    ax.set_ylabel("Progression rate", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")
    
    plt.savefig(filename + '_Progression.png')
    plt.close()

def make_plot(dpath):
    base = dpath
    data_path = base + '_episodes.json'

    with open(data_path) as json_file:
        data = json.load(json_file)

    limit = 10000

    vals = data['success'][0:limit]
    i = 0
    x = [i+1 for i in range(0, len(vals))]
    filename = base + '_plt.png'

    plot_finish_rate(x, vals, base, avg_n=500)
    vals = data['progression'][0:limit]
    plot_progression(x, vals, base, avg_n=500)
    vals = data['scores'][0:limit]
    plot_scores(x, vals, base, avg_n=500)

def plot_compare(dpath, dpath2):
    base = dpath
    data_path = base + '_episodes.json'

    with open(data_path) as json_file:
        data = json.load(json_file)

    limit = 13000

    vals = data['success'][0:limit]
    i = 0
    x = [i+1 for i in range(0, len(vals))]
    filename = base + '_plt.png'

    """ plot_finish_rate(x, vals, base, avg_n=500)
    vals = data['progression'][0:limit]
    plot_progression(x, vals, base, avg_n=500)
    vals = data['scores'][0:limit]
    plot_scores(x, vals, base, avg_n=500)"""

    base = dpath2
    data_path = base + '_episodes.json'

    with open(data_path) as json_file:
        data2 = json.load(json_file)

    vals2 = data2['success'][0:limit]
    
    plot_finish_rate_compare(x, vals, vals2, base, avg_n=500)
    """vals = data['progression'][0:limit]
    plot_progression(x, vals, base, avg_n=500)
    vals = data['scores'][0:limit]
    plot_scores(x, vals, base, avg_n=500)"""




#make_plot('/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_up120_m100k_lr0.0001_ed0.99995/mario_DDQN_up120_m100k_lr0.0001_ed0.99995')
#make_plot('/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_m100k_lr0.00005_ed0.99995-2/mario_DDQN_m100k_lr0.00005_ed0.99995-2')
#make_plot('/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_m100k_lr0.0001_ed0.99995-2/mario_DDQN_m100k_lr0.0001_ed0.99995-2')
#make_plot('/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_m100k_lr0.001_ed0.99995/mario_DDQN_m100k_lr0.001_ed0.99995')
#make_plot('/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_testingDQN_m50k_ed0.99995/mario_DDQN_testingDQN_m50k_ed0.99995')
plot_compare('/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_m100k_lr0.00005_ed0.99995-2/mario_DDQN_m100k_lr0.00005_ed0.99995-2',
            '/work/henriksv/code/mario-data-collection/mario_dqn/mario_DDQN_m100k_lr0.0001_ed0.99995-2/mario_DDQN_m100k_lr0.0001_ed0.99995-2')
"""
   episode_history = {
                'epsilon' : epsilon_history,
                'scores' : scores,
                'success' : success_history,
                'position' : pos_history,
                'time' : time_history,
                'progression' : progression_history
            }"""