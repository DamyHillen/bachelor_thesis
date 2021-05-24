from PerceptiveInferenceAgent import PerceptiveInferenceAgent
from GenerativeLayer import GenerativeLayer
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pickle
import sys
import os


COMMAND_LINE = sys.argv[0] == "main.py"


# Process parameters
HOUR_LEN = 1
DAY_LEN = 10 * HOUR_LEN
YEAR_LEN = 10 * DAY_LEN

# Simulation parameters
N_ITER = YEAR_LEN*50

# Agent parameters
LAYER_STATES = [1, 100]  # Immediately also determines number of layers


def main():
    # Creating the generative process and perceptive inference agent
    process = create_process(with_warming=True)
    agent = create_agent()

    # Lists to store the simulated values in
    generated_temps = []
    predictions = []
    agent_params = []

    # Simulation loop
    for t in range(N_ITER):
        generated_temp = process.sample(t)
        prediction = agent.predict()
        agent.update(generated_temp[0][0]["value"], prediction["layer_contributions"])

        generated_temps.append(generated_temp)
        predictions.append(prediction["value"])
        agent_params.append(agent.get_model_params())

        if t % (N_ITER/10) == 0:
            percentage = "{:.0f}".format(t // (N_ITER/100))
            print("{}{}% done".format(" "*(3 - len(percentage)), percentage))

    print("100% done")

    # Write results to disk (results/dd-mm-yy_hh:mm:ss.txt)
    store_results(generated_temps, predictions, agent_params)

    # Plot the results
    # if not COMMAND_LINE:
    #     plot_simulation(generated_temps, predictions, agent_params)


def create_process(with_warming=False):
    warming = GenerativeLayer(cycle_time=0, amplitude=0, equilibrium=10)
    # year = GenerativeLayer(parent=warming if with_warming else None, cycle_time=YEAR_LEN, amplitude=20, sigma=2.5, equilibrium=0 if with_warming else 10)
    year = GenerativeLayer(parent=warming if with_warming else None, cycle_time=YEAR_LEN, amplitude=20, sigma=0, equilibrium=0 if with_warming else 10)
    # day = GenerativeLayer(parent=year, cycle_time=DAY_LEN, offset=-DAY_LEN / 4, amplitude=10, sigma=1)
    day = GenerativeLayer(parent=year, cycle_time=DAY_LEN, offset=-DAY_LEN / 4, amplitude=10, sigma=0)
    # hour = GenerativeLayer(parent=day, cycle_time=HOUR_LEN, amplitude=0, sigma=0.25)

    return day


def create_agent():
    agent = PerceptiveInferenceAgent(layer_states=LAYER_STATES)
    return agent


def store_results(observations, predictions, agent_params):
    print("Storing results...")

    if not os.path.isdir("results"):
        os.mkdir("results")

    file = open(datetime.datetime.now().strftime("results/%d-%m-%y_%H:%M:%S::%f.results"), "wb")
    pickle.dump((HOUR_LEN, DAY_LEN, YEAR_LEN, N_ITER, LAYER_STATES, observations, predictions, agent_params), file)
    file.close()

    print("Done!")


def plot_simulation(generated_temps, predictions, agent_params):
    print("Generating plots...")

    observations = [o[0][0]["value"] for o in generated_temps]

    # General time vector
    time = list(range(N_ITER))

    # Plotting the first year of generated and predicted temperatures
    year_time = list(range(YEAR_LEN))
    plot_temperatures(year_time,
                      observations[:YEAR_LEN], "observations",
                      predictions[:YEAR_LEN], "predictions",
                      "First year generated VS predicted temperatures")

    # Plotting the last year of generated and predicted temperatures
    plot_temperatures(year_time,
                      observations[-YEAR_LEN:], "observations",
                      predictions[-YEAR_LEN:], "predictions",
                      "Last year generated VS predicted temperatures")

    plot_states(agent_params)

    print("Done!")


def plot_temperatures(time, obs, obs_label, pred, pred_label, title):
    plt.scatter(time, obs, color='k', s=10, label=obs_label)
    plt.scatter(time, pred, color='r', s=10, label=pred_label)
    plt.title(title)
    plt.xlabel("Time (iterations)")
    plt.ylabel("Temperature value")
    plt.legend()
    plt.show()


def plot_states(agent_params):
    params_per_layer = [l for l in zip(*agent_params)]

    for layer, params in enumerate(params_per_layer):
        final_params = params[-1]

        fig, axs = plt.subplots(1, len(final_params), figsize=(10, 5))

        distributions = [np.random.normal(loc=p["mu"], scale=p["sigma"], size=10000) for p in final_params]
        lower = min([min(dist) for dist in distributions])
        upper = max([max(dist) for dist in distributions])

        for state, state_params in enumerate(final_params):
            axs[state].hist(distributions[state],
                            orientation="horizontal",
                            density=True,
                            color='k',
                            bins=30,
                            range=(lower, upper))
            axs[state].set_title("State {}".format(state))
            axs[state].axhline(distributions[state].mean(), color='r')
        fig.supxlabel("Probability")
        fig.supylabel("Value")
        fig.suptitle("State parameters for layer {}".format(layer))
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
