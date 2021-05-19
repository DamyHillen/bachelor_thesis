import matplotlib.pyplot as plt
import pickle
import sys

filename = "results/19-05-21_23:59:27::650809.results"

file = open(sys.argv[1] if len(sys.argv) > 1 else filename, "rb")
HOUR_LEN, DAY_LEN, YEAR_LEN, N_ITER, LAYER_STATES, observations, predictions, agent_params = pickle.load(file)

layer0, layer1 = agent_params[-1]

points = []
for i in range(len(layer0)):
    for j in range(len(layer1)):
        points.append(layer0[i]["mu"] + layer1[j]["mu"])

plt.plot(range(len(points)), points)
plt.show()
