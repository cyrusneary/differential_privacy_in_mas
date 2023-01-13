import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

N = 5
menMeans = (20, 35, 30, 35, 27)
menStd =   (2, 3, 4, 1, 2)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)
rects1 = ax.bar(ind, menMeans, width, color='royalblue', yerr=menStd)

womenMeans = (25, 32, 34, 20, 25)
womenStd =   (3, 5, 2, 3, 3)
ax.bar(ind + 1.2 * width, menMeans, width, color='royalblue', yerr=menStd)
rects2 = ax.bar(ind + 1.2*width, womenMeans, width, bottom=menMeans, color='seagreen', yerr=womenStd)

# add some
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks((ind + 1.2*width) / 2)
ax.set_xticklabels( ('G1', 'G2', 'G3', 'G4', 'G5') )

# ax.legend( (rects1[0], rects2[0]), ('Men', 'Women') )

tikzplotlib.save('test_bar_plot.tex')

plt.show()