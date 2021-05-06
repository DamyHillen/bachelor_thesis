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
    plt.scatter(time, observations, color="k", s=10, label="observations")
    plt.scatter(time, predictions, color="r", s=10, label="predictions")
    plt.title("Generated VS predicted temperatures")
    plt.xlabel("Time (iterations)")
    plt.ylabel("Temperature value")
    plt.legend()
    plt.show()

    # Plotting the last year
    last_year_time = list(range(YEAR_LEN))
    plt.scatter(last_year_time, observations[-YEAR_LEN:], color="k", s=10, label="observations")
    plt.scatter(last_year_time, predictions[-YEAR_LEN:], color="r", s=10, label="predictions")
    plt.title("Last year generated VS predicted temperatures")
    plt.xlabel("Time (iterations)")
    plt.ylabel("Temperature value")
    plt.legend()
    plt.show()

    # Plotting the agent parameters over time
    mus, sigmas = zip(*agent_params)
    plt.plot(time, mus, color="k", label="Agent μ")
    plt.plot(time, sigmas, color="r", label="Agent σ")
    plt.title("Agent parameters over time")
    plt.xlabel("Time (iterations)")
    plt.ylabel("Parameter value")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
