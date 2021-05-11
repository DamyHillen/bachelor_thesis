from PerceptiveInferenceAgent import PerceptiveInferenceAgent
from GenerativeLayer import GenerativeLayer
import matplotlib.pyplot as plt

# Simulation parameters
N_ITER = 1000

# Process parameters
HOUR_LEN = 1
DAY_LEN = 9 * HOUR_LEN
YEAR_LEN = 9 * DAY_LEN

# Agent parameters
N_SECT = 15
SECT_SIZE = 1
N_LAYERS = 3


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
        agent.update(observation[0]["value"])
        prediction = agent.predict()

        observations.append(observation[0]["value"])
        predictions.append(prediction)
        agent_params.append(agent.get_model_params())

    plot_simulation(observations, predictions, agent_params)


def create_process():
    # warming = GenerativeLayer(cycle_time=0, amplitude=0.001, equilibrium=10)
    year = GenerativeLayer(parent=None, cycle_time=YEAR_LEN, amplitude=15, sigma=2.5)
    day = GenerativeLayer(parent=year, cycle_time=DAY_LEN, offset=-DAY_LEN / 4, amplitude=2, sigma=1)
    hour = GenerativeLayer(parent=day, cycle_time=HOUR_LEN, amplitude=0, sigma=0.25)
    return hour


def create_agent():
    agent = PerceptiveInferenceAgent(n_sect=N_SECT, sect_size=SECT_SIZE, n_layers=N_LAYERS)
    return agent


def plot_simulation(observations, predictions, agent_params):
    # General time vector
    time = list(range(N_ITER))

    # Plotting the generated and predicted temperatures
    plot_temperatures(time,
                      observations, "observations",
                      predictions, "predictions",
                      "Generated VS predicted temperatures")

    # Plotting the last year of generated and predicted temperatures
    last_year_time = list(range(YEAR_LEN))
    plot_temperatures(last_year_time,
                      observations[-YEAR_LEN:], "observations",
                      predictions[-YEAR_LEN:], "predictions",
                      "Last year generated VS predicted temperatures")

    plot_agent_params(time, agent_params)


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

    fig, axs = plt.subplots(len(params_per_layer), 1)

    for layer, layer_params in enumerate(params_per_layer):
        mus, sigmas = zip(*layer_params)
        axs[layer].plot(time, mus, color="k", label="Agent μ")
        axs[layer].plot(time, sigmas, color="r", label="Agent σ")
        axs[layer].set_title("Layer {}".format(layer))
        plt.xlabel("Time (iterations)")
        plt.ylabel("Parameter value")
        plt.legend()
    fig.suptitle("Agent parameters over time, per layer")
    plt.show()


if __name__ == "__main__":
    main()
