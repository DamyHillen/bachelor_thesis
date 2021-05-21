import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import sys


filename = "results/21-05-21_15:52:26::906897.results"

print("Loading results from '{}'...".format(filename))
file = open(sys.argv[0] if len(sys.argv) > 1 else filename, "rb")
HOUR_LEN, DAY_LEN, YEAR_LEN, N_ITER, LAYER_STATES, generated_temps, predictions, agent_params = pickle.load(file)
print("Done!")

# TODO: Make sure the agent's states are aligned properly with those of the generative process
def main():
    KLs0, KLs1 = zip(*[(KLDiv_normal(generated_temps[t][1][0], get_params(agent_params, t)[0]),
                       KLDiv_normal(generated_temps[t][1][1], get_params(agent_params, t)[1]))
                     for t in range(N_ITER)])
    time = np.arange(0, N_ITER)/YEAR_LEN

    sns.scatterplot(x=time, y=KLs1, color="blue", label="Layer 1")
    sns.scatterplot(x=time, y=KLs0, color="red", label="Layer 0")
    plt.title("KL Divergence of the generative process and generative model")
    plt.xlabel("Time (years)")
    plt.ylabel("KL Divergence")
    plt.show()

    print("Done!")


def get_params(params, t):
    state0 = (t + 1) % LAYER_STATES[0]
    state1 = ((t + 1) // LAYER_STATES[0]) % LAYER_STATES[1]
    return params[t][0][state0], params[t][1][state1]


def KLDiv_normal(params0, params1):
    mu0, sigma0 = params0["mu"], params0["sigma"]
    mu1, sigma1 = params1["mu"], params1["sigma"]

    return np.log(sigma1/sigma0) + (np.square(sigma0) + np.square(mu0 - mu1))/(2 * np.square(sigma1)) - 1/2


if __name__ == "__main__":
    main()
