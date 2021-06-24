import numpy as np


class PerceptiveInferenceAgent:
    def __init__(self, layer_states, prior=None, only_mu=False):
        self.model = ModelLayer(layer_states=layer_states, prior=prior)
        self.prior = prior.copy()
        self.only_mu = only_mu

    def update(self, obs, layer_contributions):
        self.model.update(obs, layer_contributions)

    def predict(self):
        return self.model.predict(only_mu=self.only_mu)

    def get_model_params(self):
        return self.model.get_params()


class ModelLayer:
    def __init__(self, transition_speed=1, layer_states=[], prior=None):
        self.parent = ModelLayer(transition_speed=layer_states[0],
                                 layer_states=layer_states[1:],
                                 prior=prior) if len(layer_states) > 1 else None

        self.n_states = layer_states[0]
        self.state_variables = [{"n": 5,
                                 "mu": 0,
                                 "sigma": 1} if not prior else prior.copy() for _ in range(self.n_states)]
        self.in_state = 0
        self.transition_speed = transition_speed
        self.state_transition = -1
        self.first = True

    def update(self, obs, layer_contributions):
        self.state_variables[self.in_state]["n"] += 1

        old_mu = self.state_variables[self.in_state]["mu"]
        old_sigma = self.state_variables[self.in_state]["sigma"]
        n = self.state_variables[self.in_state]["n"]

        # contribution = layer_contributions[0]
        should_have_been = obs - (layer_contributions[1] if self.parent else 0)
        # error = contribution - should_have_been

        # Incrementally update average and standard deviation
        self.state_variables[self.in_state]["mu"] = (old_mu * (n-1) + should_have_been)/n
        self.state_variables[self.in_state]["sigma"] = np.sqrt(((n - 2) / (n - 1)) * np.power(old_sigma, 2) + np.power((should_have_been - old_mu), 2) / n)

        if self.parent:
            self.parent.update(obs-self.state_variables[self.in_state]["mu"], layer_contributions[1:])

    def predict(self, prediction=None, only_mu=False):
        self.state_transition = (self.state_transition + 1) % self.transition_speed
        if self.state_transition == 0:
            if self.first:
                self.first = False
            else:
                self.in_state = (self.in_state + 1) % self.n_states

        if not prediction:
            prediction = {"layer_contributions": [], "value": 0}

        layer_contribution = np.random.normal(loc=self.state_variables[self.in_state]["mu"],
                                              scale=(0 if only_mu else self.state_variables[self.in_state]["sigma"]))

        prediction["layer_contributions"].append(layer_contribution)
        prediction["value"] += layer_contribution

        if self.parent:
            self.parent.predict(prediction, only_mu=only_mu)

        return prediction

    def get_params(self):
        return [[vars.copy() for vars in self.state_variables]] + (self.parent.get_params() if self.parent else [])
