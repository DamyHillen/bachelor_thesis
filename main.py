from PerceptiveInferenceAgent import PerceptiveInferenceAgent
from GenerativeLayer import GenerativeLayer
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os


# Process parameters
HOUR_LEN = 1
DAY_LEN = 10 * HOUR_LEN
YEAR_LEN = 10 * DAY_LEN

# Simulation parameters
N_ITER = YEAR_LEN*5000

# Agent parameters
LAYER_STATES = [10, 10]  # Immediately also determines number of layers


def main():
    # Creating the generative process and perceptive inference agent
    process = create_process()
    agent = create_agent()

    observations = []
    predictions = []
    agent_params = []

    # Simulation loop
    for t in range(N_ITER):
        observation = process.sample(t)
        prediction = agent.predict()
        agent.update(observation[0]["value"], prediction["layer_contributions"])

        observations.append(observation[0]["value"])
        predictions.append(prediction["value"])
        agent_params.append(agent.get_model_params())

    # Store results to disk (results/dd-mm-yy_hh:mm:ss.txt)
    store_results(observations, predictions, agent_params)

    # Plot the results
    plot_simulation(observations, predictions, agent_params)


def create_process():
    # warming = GenerativeLayer(cycle_time=0, amplitude=0.001, equilibrium=10)
    year = GenerativeLayer(parent=None, cycle_time=YEAR_LEN, amplitude=20, sigma=2.5, equilibrium=10)
    day = GenerativeLayer(parent=year, cycle_time=DAY_LEN, offset=-DAY_LEN / 4, amplitude=10, sigma=1)
    # hour = GenerativeLayer(parent=day, cycle_time=HOUR_LEN, amplitude=0, sigma=0.25)

    return day


def create_agent():
    agent = PerceptiveInferenceAgent(layer_states=LAYER_STATES)
    return agent


def store_results(observations, predictions, agent_params):
    if not os.path.isdir("results"):
        os.mkdir("results")

    file = open(datetime.datetime.now().strftime("results/%d-%m-%y_%H:%M:%S.txt"), "w")
    file.write("HOUR_LEN, DAY_LEN, YEAR_LEN: {}\n".format(repr([HOUR_LEN, DAY_LEN, YEAR_LEN])))
    file.write("N_ITER: {}\n".format(N_ITER))
    file.write("LAYER_STATES: {}\n".format(repr(LAYER_STATES)))
    file.write("observations: {}\n".format(repr(observations)))
    file.write("predictions: {}\n".format(repr(predictions)))
    file.write("agent_params: {}\n".format(repr(agent_params)))
    file.close()


def plot_simulation(observations, predictions, agent_params):
    # General time vector
    time = list(range(N_ITER))

    # Plotting the generated and predicted temperatures
    # plot_temperatures(time,
    #                   observations, "observations",
    #                   predictions, "predictions",
    #                   "Generated VS predicted temperatures")

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
