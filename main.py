import multiprocessing

from PerceptiveInferenceAgent import PerceptiveInferenceAgent
from GenerativeLayer import GenerativeLayer
from multiprocessing import Pool
from Simulation import Simulation
import numpy as np
import datetime
import pickle
import os

N_PROCESSES = 8
N_SIMS = 50

# Process parameters
HOUR_LEN = 1
DAY_LEN = 10 * HOUR_LEN
YEAR_LEN = 10 * DAY_LEN

# Simulation parameters
N_ITER = YEAR_LEN*500

# Agent parameters
LAYER_STATES = [1, 1, 10, 10]  # Immediately also determines number of layers


def main():
    # Creating the generative process and perceptive inference agent
    # process = create_process(with_warming=False)
    # agent = create_agent(prior={"n": 5, "mu": 10, "sigma": 1}, only_mu=False)

    sims = [Simulation(id=s,
                       agent=create_agent(prior={"n": 5, "mu": np.random.uniform(-20, 20), "sigma": 1}),
                       process=create_process(),
                       n_iter=N_ITER,
                       layer_states=LAYER_STATES) for s in range(N_SIMS)]

    with Pool(processes=N_PROCESSES) as pool:
        model_errors = pool.map(run_sim, sims)

    # generated_temps = sim.generated_temps
    # predictions = sim.predictions
    # agent_params = sim.agent_params
    # model_errors = [sim.model_errors for sim in sims]

    store_model_err(model_errors)

    # Write results to disk (results/dd-mm-yy_hh:mm:ss.txt) # store_results(generated_temps, predictions, agent_params)


def run_sim(sim):
    return sim.run()


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


def store_model_err(model_errors):
    # print("Storing errors...")

    if not os.path.isdir("results/errors"):
        os.mkdir("results/errors")

    file = open(datetime.datetime.now().strftime("results/errors/%d-%m-%y_%H:%M:%S::%f.errors"), "wb")
    pickle.dump((N_ITER, YEAR_LEN, model_errors), file)
    file.close()

    print("Done!")


if __name__ == "__main__":
    main()
