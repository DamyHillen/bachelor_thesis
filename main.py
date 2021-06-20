from PerceptiveInferenceAgent import PerceptiveInferenceAgent
from GenerativeLayer import GenerativeLayer
from multiprocessing import Pool
from Simulation import Simulation
import re as regex
import numpy as np
import msgpack
import time
import os

N_PROCESSES = 10
N_SIMS = 1

# Process parameters
HOUR_LEN = 1
DAY_LEN = 10 * HOUR_LEN
YEAR_LEN = 10 * DAY_LEN

# Simulation parameters
WITH_WARMING = False
WITH_YEAR = False
N_ITER = YEAR_LEN * 150

# Agent parameters
ONLY_MU = True
LAYER_STATES = [11]  # Immediately also determines number of layers


class Main:
    def __init__(self):
        pass

    def start(self):
        t = time.time()

        priors = np.random.uniform(-100, 100, N_SIMS)
        priors = [0]

        sims = [Simulation(id=s,
                           agent=self.create_agent(prior={"n": 3, "mu": priors[s], "sigma": 1}, only_mu=ONLY_MU),
                           process=self.create_process(with_year=WITH_YEAR, with_warming=WITH_WARMING),
                           n_iter=N_ITER,
                           layer_states=LAYER_STATES) for s in range(N_SIMS)]

        with Pool(processes=N_PROCESSES) as pool:
            sim_results = pool.map(self.run_sim, sims)

        print("{:.2f} seconds".format(time.time() - t))

        # self.store_model_err(sim_results)
        # self.store_state_results(sim_results)
        self.store_single_result(sim_results[0][0], sim_results[0][1], sim_results[0][2], sim_results[0][3])

    @staticmethod
    def run_sim(sim):
        return sim.run()

    @staticmethod
    def create_process(with_year=True, with_warming=False):
        warming = GenerativeLayer(cycle_time=0, amplitude=0, equilibrium=10)
        year = GenerativeLayer(parent=warming if with_warming else None, cycle_time=YEAR_LEN, amplitude=20, sigma=0, equilibrium=0 if with_warming else 10)
        day = GenerativeLayer(parent=year if with_year else None, cycle_time=DAY_LEN, offset=-DAY_LEN / 4, amplitude=10, sigma=0, equilibrium=0 if with_year else 10)

        return day

    @staticmethod
    def create_agent(prior=None, only_mu=False):
        agent = PerceptiveInferenceAgent(layer_states=LAYER_STATES, prior=prior, only_mu=only_mu)
        return agent

    @staticmethod
    def store_single_result(prior, generated_temps, predictions, agent_params):
        print("Storing result...")
        t = time.time()

        directory = "results/single/"
        if not os.path.isdir(directory):
            os.mkdir(directory)

        filename = directory + regex.sub(", ", "-", "{}_{}y_{}a{}.single".format(LAYER_STATES, N_ITER//YEAR_LEN, N_SIMS, "_mu" if ONLY_MU else ""))
        with open(filename, "wb") as file:
            file.write(msgpack.packb({"HOUR_LEN": HOUR_LEN,
                                      "DAY_LEN": DAY_LEN,
                                      "YEAR_LEN": YEAR_LEN,
                                      "N_ITER": N_ITER,
                                      "LAYER_STATES": LAYER_STATES,
                                      "prior": prior,
                                      "generated_temps": generated_temps,
                                      "predictions": predictions,
                                      "agent_params": agent_params}))
            file.close()

        print("Written to {}".format(filename))
        print("Done! ({:.2f} seconds)".format(time.time() - t))

    @staticmethod
    def store_state_results(agent_params):
        print("Storing states...")
        t = time.time()

        directory = "results/states/"
        if not os.path.isdir(directory):
            os.mkdir(directory)

        filename = directory + regex.sub(", ", "-", "{}_{}y_{}a{}.states".format(LAYER_STATES, N_ITER//YEAR_LEN, N_SIMS, "_mu" if ONLY_MU else ""))
        with open(filename, "wb") as file:
            file.write(msgpack.packb({"HOUR_LEN": HOUR_LEN,
                                      "DAY_LEN": DAY_LEN,
                                      "YEAR_LEN": YEAR_LEN,
                                      "N_ITER": N_ITER,
                                      "LAYER_STATES": LAYER_STATES,
                                      "agent_params": agent_params}))
            file.close()

        print("Written to {}".format(filename))
        print("Done! ({:.2f} seconds)".format(time.time() - t))

    @staticmethod
    def store_model_err(model_errors):
        print("Storing errors...")
        t = time.time()

        directory = "results/errors/"
        if not os.path.isdir(directory):
            os.mkdir(directory)

        filename = directory + regex.sub(", ", "-", "{}_{}y_{}a{}.errors".format(LAYER_STATES, N_ITER // YEAR_LEN, N_SIMS, "_mu" if ONLY_MU else ""))
        with open(filename, "wb") as file:
            file.write(msgpack.packb({"HOUR_LEN": HOUR_LEN,
                                      "DAY_LEN": DAY_LEN,
                                      "YEAR_LEN": YEAR_LEN,
                                      "N_ITER": N_ITER,
                                      "LAYER_STATES": LAYER_STATES,
                                      "model_errors": model_errors}))
            file.close()

        print("Written to {}".format(filename))
        print("Done! ({:.2f} seconds)".format(time.time() - t))


if __name__ == "__main__":
    Main().start()
