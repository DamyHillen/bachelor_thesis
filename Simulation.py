from tqdm import tqdm
import sys

class Simulation:
    def __init__(self, id, agent, process, n_iter, layer_states):
        self.id = id
        # self.pbar = tqdm(total=n_iter)
        # self.pbar.set_description("Simulation [{}]".format(id))

        self.agent = agent
        self.process = process
        self.n_iter = n_iter
        self.layer_states = layer_states

        # Simulation values:
        self.generated_temps = []
        self.predictions = []
        self.agent_params = []
        self.model_errors = []

    def run(self):
        # Lists to store the simulated values in

        # Simulation loop
        for t in range(self.n_iter):
            # self.pbar.update(1)

            generated_temp = self.process.sample(t)
            prediction = self.agent.predict()
            self.agent.update(generated_temp[0][0]["value"], prediction["layer_contributions"])

            self.generated_temps.append(generated_temp)
            self.predictions.append(prediction["value"])
            self.agent_params.append(self.agent.get_model_params())
            self.model_errors.append(abs(generated_temp[0][0]["value"] - self.get_mu_sum(self.agent_params, t)))

        print("Simulation [{}] done!".format(self.id), file=sys.stderr)
        return self.model_errors

    def get_mu_sum(self, params, t):
        state0 = t % self.layer_states[0]
        states = [state0]
        for i in range(1, len(params[t])):
            states.append((t // self.layer_states[i-1]) % self.layer_states[i])

        return sum([params[t][i][states[i]]["mu"] for i in range(len(params[t]))])
