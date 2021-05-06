import numpy as np


class GenerativeLayer:
    def __init__(self, parent=None, cycle_time=1, offset=0, amplitude=1, equilibrium=0, sigma=0):
        self.parent = parent            # The parent that determines the equilibrium/intercept of this layer.
        self.cycle_time = cycle_time    # > 0 for oscillation, = 0 for linear
        self.offset = offset            # Time offset
        self.amplitude = amplitude      # Amplitude of oscillation, slope of linear
        self.equilibrium = equilibrium  # Equilibrium/intercept of the resulting values
        self.sigma = sigma              # Standard deviation of the sampled distribution

        self.history = {}               # Keeps track of sampled values so far

    def sample(self, t):
        parent_history = []
        if self.parent:
            parent_history = self.parent.sample(t)
            self.equilibrium = parent_history[0]["value"]

        if t not in self.history:
            self.history[t] = {"value": 0,        # Actual value
                               "sampled": 0,      # Sampled value without equilibrium
                               "cycle_noise": 0}  # Noise value of the cycle

        if self.cycle_time > 0:  # Oscillation
            if t % self.cycle_time == 0:  # New cycle
                self.history[t]["cycle_noise"] = self.new_noise()
            else:  # Same cycle as previous time step
                self.history[t]["cycle_noise"] = self.history[t - 1]["cycle_noise"]

        self.history[t]["value"] = self.new_val(t)

        return [self.history[t]] + parent_history

    def new_val(self, t):
        if self.cycle_time > 0:  # Oscillation
            self.history[t]["sampled"] = self.amplitude * np.sin((t + self.offset) * 2 * np.pi / self.cycle_time) \
                                       + self.history[t]["cycle_noise"]
            return self.equilibrium + self.history[t]["sampled"]

        # Linear
        self.history[t]["sampled"] = self.amplitude * (t + self.offset) + self.history[t]["cycle_noise"]
        return self.equilibrium + self.history[t]["sampled"]

    def new_noise(self):
        return np.random.normal(scale=self.sigma)
