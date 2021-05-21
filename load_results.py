import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import sys


filename = "results/20-05-21_14:49:50::536230.results"

print("Loading results from '{}'...".format(filename))
file = open(sys.argv[0] if len(sys.argv) > 1 else filename, "rb")
HOUR_LEN, DAY_LEN, YEAR_LEN, N_ITER, LAYER_STATES, observations, predictions, agent_params = pickle.load(file)
print("Done!")


def main():
    # MSEs = [MSE(predictions[t], gen_prediction_dist(agent_params[t], t)) for t in range(N_ITER)]
    mu_MSEs = [MSE(predictions[t], get_mu(agent_params[t], t)) for t in range(N_ITER)]

    window = 50

    avg = np.convolve(mu_MSEs, np.ones(window), 'valid') / window
    time = np.arange(0, N_ITER)/YEAR_LEN
    print("100% done")

    sns.scatterplot(x=time, y=mu_MSEs, color="black")
    sns.lineplot(x=time[window-1:], y=avg, color="red")
    plt.show()

    print("Done!")


def get_mu(params, t):
    if t % (N_ITER / 10) == 0:
        percentage = "{:.0f}".format(t // (N_ITER / 100))
        print("{}{}% done".format(" " * (3 - len(percentage)), percentage))

    state0 = (t + 1) % LAYER_STATES[0]
    state1 = ((t + 1) // LAYER_STATES[0]) % LAYER_STATES[1]

    return params[0][state0]["mu"] + params[1][state1]["mu"]


def gen_prediction_dist(params, t):
    if t % (N_ITER / 10) == 0:
        percentage = "{:.0f}".format(t // (N_ITER / 100))
        print("{}{}% done".format(" " * (3 - len(percentage)), percentage))

    state0 = (t+1) % LAYER_STATES[0]
    state1 = ((t+1) // LAYER_STATES[0]) % LAYER_STATES[1]
    dist = np.random.normal(loc=np.random.normal(loc=params[1][state1]["mu"],
                                                 scale=params[1][state1]["sigma"])
                                                     + params[0][state0]["mu"],
                            scale=params[0][state0]["sigma"],
                            size=500)
    return dist


def MSE(x, p):
    p = np.array(p)  # Ensure p is a numpy array
    return np.mean(np.power(p - x, 2))


if __name__ == "__main__":
    main()
