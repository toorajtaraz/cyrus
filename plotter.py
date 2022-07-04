# Read the file res.txt which has this format
# <time in ms> <width> <height>
# and plot the data in a bar chart 
# add legends for each bar and show the width x height 

import matplotlib.pyplot as plt

file_path = ".\\res.txt"
info = []

with open(file_path, "r") as f:
    for line in f:
        res = line.split()
        res = [int(x) for x in res]
        res = tuple(res)
        info.append(res)

# plot based on first element in each tuple and show the next two as legend and show all ticks
plt.bar(range(len(info)), [x[0] for x in info], align="center")
plt.xticks(range(len(info)), [str(str(x[1]) + " X " + str(x[2])) for x in info])
for index, value in enumerate(info):
    plt.text(index, value[0], str(value[0]), ha="center", va="bottom")
plt.show()
