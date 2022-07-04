# Read the file res.txt which has this format
# <time in ms> <width> <height>
# and plot the data in a bar chart 
# add legends for each bar and show the width x height 

import matplotlib.pyplot as plt

file_path = ".\\res.txt"
info = []

with open(file_path, "r") as f:
    for line in f:
        info.append(line.split())

print(info)