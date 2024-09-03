#! /bin/python3

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas

data = pandas.read_csv('results.csv')

mpl.rcParams["savefig.facecolor"] = "white"
mpl.rcParams["axes.facecolor"] = "white"
plt.style.use('fivethirtyeight')

plt.figure(figsize=(15, 10))

offload = [d * 100 for d in data["offload"]]
runtime = [d / 1000 for d in data["runtime"]]

plt.title('Runtime in Regards to GPU Offload (lower is better) - 100B Iterations')
plt.xlabel("GPU Offload Percentage (%)")
plt.ylabel("Runtime (s)")

plt.plot(offload, runtime, marker='o', linestyle='-')
plt.grid(True)

plt.savefig("results.png")
