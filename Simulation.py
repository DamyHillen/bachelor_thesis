import sys


class Simulation:
    def __init__(self, id, agent, process, n_iter, layer_states):
        self.id = id

        self.agent = agent
        self.process = process
        self.n_iter = n_iter
        self.layer_states = layer_states

        # Simulation values:
        self.generated_temps = []
        self.predictions = []
        self.agent_params = []

        self.eqs = []
        self.amps = []
        self.sigmas = []

        # self.model_errors = []

    def run(self):
        # Lists to store the simulated values in

        # Simulation loop
        for t in range(self.n_iter):

            generated_temp = self.process.sample(t)
            prediction = self.agent.predict()
            self.agent.update(generated_temp[0][0]["value"], prediction["layer_contributions"])

            # self.generated_temps.append(generated_temp)
            # self.predictions.append(prediction["value"])
            # self.agent_params.append(self.agent.get_model_params())

            params = self.agent.get_model_params()
            mus_per_layer = [[p["mu"] for p in l] for l in params]
            self.amps.append([(max(mus) - min(mus))/2 for mus in mus_per_layer])
            self.eqs.append([min(mus) + self.amps[-1][i] for i, mus in enumerate(mus_per_layer)])
            current_params = self.get_params(params, t)
            self.sigmas.append([p["sigma"] for p in current_params])
            # self.model_errors.append(abs(generated_temp[0][0]["value"] - self.get_mu_sum(self.agent.get_model_params(), t)))

        print("Simulation [{}] done!".format(self.id), file=sys.stderr)
        # return self.model_errors
        return [self.amps, self.eqs, self.sigmas]
        # return [self.agent.prior, self.generated_temps, self.predictions, self.agent_params]

    def get_mu_sum(self, params, t):
        state0 = t % self.layer_states[0]
        states = [state0]
        for i in range(1, len(params)):
            states.append((t // self.layer_states[i-1]) % self.layer_states[i])

        return sum([params[i][states[i]]["mu"] for i in range(len(params))])

    def get_params(self, params, t):
        state0 = t % self.layer_states[0]
        states = [state0]
        for i in range(1, len(params)):
            states.append((t // self.layer_states[i-1]) % self.layer_states[i])

        return [params[i][states[i]] for i in range(len(params))]
