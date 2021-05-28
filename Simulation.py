from tqdm import tqdm


class Simulation:
    def __init__(self, id, agent, process, n_iter):
        self.id = id
        self.pbar = tqdm(total=n_iter)
        self.pbar.set_description("Simulation [{}]".format(id))

        self.agent = agent
        self.process = process
        self.n_iter = n_iter

        # Simulation values:
        self.generated_temps = []
        self.predictions = []
        self.agent_params = []

    def run(self):
        # Lists to store the simulated values in

        # Simulation loop
        for t in range(self.n_iter):
            self.pbar.update(1)

            generated_temp = self.process.sample(t)
            prediction = self.agent.predict()
            self.agent.update(generated_temp[0][0]["value"], prediction["layer_contributions"])

            self.generated_temps.append(generated_temp)
            self.predictions.append(prediction["value"])
            self.agent_params.append(self.agent.get_model_params())
