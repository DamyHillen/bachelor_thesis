from PerceptiveInferenceAgent import PerceptiveInferenceAgent
from GenerativeLayer import GenerativeLayer
from Simulation import Simulation
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import datetime
import pickle
import sys
import os


# Process parameters
HOUR_LEN = 1
DAY_LEN = 10 * HOUR_LEN
YEAR_LEN = 10 * DAY_LEN

# Simulation parameters
N_ITER = YEAR_LEN*500

# Agent parameters
LAYER_STATES = [10, 10]  # Immediately also determines number of layers


def main():
    # Creating the generative process and perceptive inference agent
    process = create_process(with_warming=False)
    agent = create_agent(prior={"n": 5, "mu": 10, "sigma": 1}, only_mu=False)

    sim = Simulation(1, agent, process, N_ITER)
    sim.run()

    generated_temps = sim.generated_temps
    predictions = sim.predictions
    agent_params = sim.agent_params

    # Write results to disk (results/dd-mm-yy_hh:mm:ss.txt)
    # store_results(generated_temps, predictions, agent_params)


def create_process(with_warming=False):
    warming = GenerativeLayer(cycle_time=0, amplitude=0, equilibrium=10)
    # year = GenerativeLayer(parent=warming if with_warming else None, cycle_time=YEAR_LEN, amplitude=20, sigma=2.5, equilibrium=0 if with_warming else 10)
    year = GenerativeLayer(parent=warming if with_warming else None, cycle_time=YEAR_LEN, amplitude=20, sigma=0, equilibrium=0 if with_warming else 10)
    # day = GenerativeLayer(parent=year, cycle_time=DAY_LEN, offset=-DAY_LEN / 4, amplitude=10, sigma=1)
    day = GenerativeLayer(parent=year, cycle_time=DAY_LEN, offset=-DAY_LEN / 4, amplitude=10, sigma=0)
    # hour = GenerativeLayer(parent=day, cycle_time=HOUR_LEN, amplitude=0, sigma=0.25)

    return day


def create_agent(prior=None, only_mu=False):
    agent = PerceptiveInferenceAgent(layer_states=LAYER_STATES, prior=prior, only_mu=only_mu)
    return agent


def store_results(observations, predictions, agent_params):
    print("Storing results...")

    if not os.path.isdir("results"):
        os.mkdir("results")

    file = open(datetime.datetime.now().strftime("results/%d-%m-%y_%H:%M:%S::%f.results"), "wb")
    pickle.dump((HOUR_LEN, DAY_LEN, YEAR_LEN, N_ITER, LAYER_STATES, observations, predictions, agent_params), file)
    file.close()

    print("Done!")


if __name__ == "__main__":
    main()
