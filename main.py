from PerceptiveInferenceAgent import PerceptiveInferenceAgent
from GenerativeLayer import GenerativeLayer
from multiprocessing import Pool
from Simulation import Simulation
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import datetime
import pickle
import time
import os

N_PROCESSES = 10
N_SIMS = 2000

# Process parameters
HOUR_LEN = 1
DAY_LEN = 10 * HOUR_LEN
YEAR_LEN = 10 * DAY_LEN

# Simulation parameters
N_ITER = YEAR_LEN*100

# Agent parameters
LAYER_STATES = [10, 10]  # Immediately also determines number of layers


class Main:
    def __init__(self):
        pass

    def start(self):
        # Creating the generative process and perceptive inference agent
        # process = create_process(with_warming=False)
        # agent = create_agent(prior={"n": 5, "mu": 10, "sigma": 1}, only_mu=False)
        t = time.time()

        priors = np.random.uniform(-100, 100, N_SIMS)

        sims = [Simulation(id=s,
                           agent=self.create_agent(prior={"n": 5, "mu": priors[s], "sigma": 1}),
                           process=self.create_process(),
                           n_iter=N_ITER,
                           layer_states=LAYER_STATES) for s in range(N_SIMS)]

        with Pool(processes=N_PROCESSES) as pool:
            model_errors = pool.map(self.run_sim, sims)
            # model_errors = pool.map_async(run_sim, sims)

        # generated_temps = sim.generated_temps
        # predictions = sim.predictions
        # agent_params = sim.agent_params
        print("{:.2f} seconds".format(time.time() - t))

        self.store_model_err(model_errors)

        # Write results to disk (results/dd-mm-yy_hh:mm:ss.txt) # store_results(generated_temps, predictions, agent_params)

    @staticmethod
    def run_sim(sim):
        return sim.run()

    @staticmethod
    def create_process(with_warming=False):
        warming = GenerativeLayer(cycle_time=0, amplitude=0, equilibrium=10)
        # year = GenerativeLayer(parent=warming if with_warming else None, cycle_time=YEAR_LEN, amplitude=20, sigma=2.5, equilibrium=0 if with_warming else 10)
        year = GenerativeLayer(parent=warming if with_warming else None, cycle_time=YEAR_LEN, amplitude=20, sigma=0, equilibrium=0 if with_warming else 10)
        # day = GenerativeLayer(parent=year, cycle_time=DAY_LEN, offset=-DAY_LEN / 4, amplitude=10, sigma=1)
        day = GenerativeLayer(parent=year, cycle_time=DAY_LEN, offset=-DAY_LEN / 4, amplitude=10, sigma=0)
        # hour = GenerativeLayer(parent=day, cycle_time=HOUR_LEN, amplitude=0, sigma=0.25)

        return day

    @staticmethod
    def create_agent(prior=None, only_mu=False):
        agent = PerceptiveInferenceAgent(layer_states=LAYER_STATES, prior=prior, only_mu=only_mu)
        return agent

    @staticmethod
    def store_results(observations, predictions, agent_params):
        print("Storing results...")

        if not os.path.isdir("results"):
            os.mkdir("results")

        file = open(datetime.datetime.now().strftime("results/%d-%m-%y_%H:%M:%S::%f.results"), "wb")
        pickle.dump((HOUR_LEN, DAY_LEN, YEAR_LEN, N_ITER, LAYER_STATES, observations, predictions, agent_params), file)
        file.close()

        print("Done!")

    @staticmethod
    def store_model_err(model_errors):
        print("Storing errors...")

        if not os.path.isdir("results/errors"):
            os.mkdir("results/errors")

        file = open(datetime.datetime.now().strftime("results/errors/%d-%m-%y_%H:%M:%S::%f.errors"), "wb")
        pickle.dump((N_ITER, YEAR_LEN, LAYER_STATES, model_errors), file)
        file.close()

        print("Done!")


if __name__ == "__main__":
    Main().start()
