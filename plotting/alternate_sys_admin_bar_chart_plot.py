import pickle
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

with open('bar_chart_data.pkl', 'rb') as f:
    results = pickle.load(f)

print(results)

width = 0.1

fig = plt.figure()
ax = fig.add_subplot(111)

bottoms_md = np.zeros((len(results.keys())-1,))
bottoms_base = np.zeros((len(results.keys())-1,))

ind = np.arange(3)

num_eps_trials = len(results[list(results.keys())[0]]['md'])

for eps_ind in range(num_eps_trials - 1):

    heights_md = []
    heights_base = []
    for trial in ['0001', '0022', '2233']:
        heights_md.append(results[trial]['md'][eps_ind])
        heights_base.append(results[trial]['base'][eps_ind])

    ax.bar(ind + eps_ind * (1.1 * width), heights_md, width)
    ax.bar(ind + eps_ind * (1.1 * width) + num_eps_trials * (1.1 * width) + width/4, heights_base, width)

xmin, xmax = ax.get_xlim()
ax.plot([xmin, xmax], [1.0, 1.0], 'k', linewidth=1.0)

# add some labels
ax.set_ylabel('Probability of Success')
ax.set_xticks(np.array([0.36, 1.35, 2.35]))
ax.set_xticklabels(['0001', '0022', '2233'])

tikzplotlib.save('tikz/sys_admin_bar_chart.tex')

# plt.show()