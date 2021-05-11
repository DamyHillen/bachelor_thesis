from PerceptiveInferenceAgent import PerceptiveInferenceAgent
from GenerativeLayer import GenerativeLayer
import matplotlib.pyplot as plt
import numpy as np


# Process parameters
HOUR_LEN = 1
DAY_LEN = 5 * HOUR_LEN
YEAR_LEN = 10 * DAY_LEN

# Simulation parameters
N_ITER = YEAR_LEN*5000

# Agent parameters
LAYER_STATES = [5, 10]  # Immediately also determines number of layers


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

    # TODO: Make this work for multi-layer, multi-state parameters
    # plot_agent_params(time, agent_params)

    plot_states(agent_params)


def plot_temperatures(time, obs, obs_label, pred, pred_label, title):
    plt.scatter(time, obs, color='k', s=10, label=obs_label)
    plt.scatter(time, pred, color='r', s=10, label=pred_label)
    plt.title(title)
    plt.xlabel("Time (iterations)")
    plt.ylabel("Temperature value")
    plt.legend()
    plt.show()


# TODO: Fix for multiple layers
def plot_agent_params(time, agent_params):
    params_per_layer = [l for l in zip(*agent_params)]

    if len(params_per_layer) > 1:
        fig, axs = plt.subplots(len(params_per_layer), 1, figsize=(10, len(params_per_layer)*2.5))

        for layer, layer_params in enumerate(params_per_layer):
            mus, sigmas = zip(*layer_params)
            axs[layer].plot(time, mus, color="k", label="Agent μ")
            axs[layer].plot(time, sigmas, color="r", label="Agent σ")
            axs[layer].set_title("Layer {} ({} state(s))".format(layer, LAYER_STATES[layer]))
            axs[layer].legend()
        fig.supxlabel("Time (iterations)")
        fig.supylabel("Parameter value")
        fig.suptitle("Agent parameters over time, per layer")
        plt.tight_layout()
        plt.show()
    else:
        mus, sigmas = zip(*params_per_layer[0])
        plt.plot(time, mus, color="k", label="Agent μ")
        plt.plot(time, sigmas, color="r", label="Agent σ")
        plt.title("Agent parameters over time")
        plt.xlabel("Time (iterations)")
        plt.ylabel("Parameter value")
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
