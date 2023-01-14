import pickle
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

with open('bar_chart_data.pkl', 'rb') as f:
    results = pickle.load(f)

print(results)

width = 0.08

fig = plt.figure()
ax = fig.add_subplot(111)

bottoms_md = np.zeros((len(results.keys()),))
bottoms_base = np.zeros((len(results.keys()),))

ind = np.arange(3)

num_eps_trials = len(results[list(results.keys())[0]]['md'])

for eps_ind in range(num_eps_trials):

    heights_md = []
    heights_base = []
    for trial in ['0001', '0022', '2233']:
        heights_md.append(results[trial]['md'][eps_ind])
        heights_base.append(results[trial]['base'][eps_ind])

    ax.bar(ind + eps_ind * (1.2 * width), heights_md, width)
    ax.bar(ind + eps_ind * (1.2 * width) + num_eps_trials * (1.2 * width) + width/2, heights_base, width)

    # ax.bar(ind + 1.2 * width, results[trial]['md'], width, bottom=bottoms_md)

    # ax.bar(ind, results[trial]['md'], width, bottom=bottoms_md)
    # ax.bar(ind + 1.2 * width, results[trial]['base'], width, bottom=bottoms_base)

    # bottoms_md = bottoms_md + results[trial]['md']
    # bottoms_base = bottoms_base + results[trial]['base']

# for i in range(len(results[list(results.keys())[0]]['md'])):
#     bar_heights_md = []
#     bar_heights_base = []

#     for trial in results.keys():
#         bar_heights_md.append(results[trial]['md'][i] - bottoms_md[i])
#         bar_heights_base.append(results[trial]['base'][i] - bottoms_base[i])

#     ax.bar(ind, bar_heights_md, width, bottom=bottoms_md)
#     ax.bar(ind + 1.2 * width, bar_heights_base, width, bottom=bottoms_base)

#     bottoms_md = bottoms_md + bar_heights_md
#     bottoms_base = bottoms_base + bar_heights_base

# add some labels
ax.set_ylabel('Probability of Success')
ax.set_xticks(np.array([0.36, 1.35, 2.35]))
ax.set_xticklabels(['0001', '0022', '2233'])

tikzplotlib.save('tikz/sys_admin_bar_chart.tex')

plt.show()


# N = 5
# menMeans = (20, 35, 30, 35, 27)
# menStd =   (2, 3, 4, 1, 2)

# ind = np.arange(N)  # the x locations for the groups
# width = 0.35       # the width of the bars


# rects1 = ax.bar(ind, menMeans, width, color='royalblue', yerr=menStd)

# womenMeans = (25, 32, 34, 20, 25)
# womenStd =   (3, 5, 2, 3, 3)
# ax.bar(ind + 1.2 * width, menMeans, width, color='royalblue', yerr=menStd)
# rects2 = ax.bar(ind + 1.2*width, womenMeans, width, bottom=menMeans, color='seagreen', yerr=womenStd)