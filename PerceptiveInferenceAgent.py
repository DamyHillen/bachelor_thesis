from collections import Counter
import numpy as np


class PerceptiveInferenceAgent:
    def __init__(self, n_sect, sect_size, n_layers):
        self.n_layers = n_layers
        self.model = ModelLayer(n_sect, sect_size, n_layers)

    def update(self, x):
        self.model.update(x)

    def predict(self):
        return self.model.predict()

    def get_model_params(self):
        return self.model.get_params()


class ModelLayer:
    def __init__(self, n_sect, sect_size, ancestor_count=0):
        self.n_sect = n_sect
        self.sect_size = sect_size
        self.sect_counts = Counter({k: 0 for k in range(n_sect)})

        self.parent = ModelLayer(n_sect, sect_size, ancestor_count-1) if ancestor_count > 1 else None

        self.n = 3  # Start at n = 3 to avoid division by 0 for sigma
        self.mu = 0
        self.sigma = 0

    # TODO: Infer current state from observation.
    # TODO: Propagate the update upwards to the parent layers.
    def update(self, x):
        self.sect_counts.update([self.get_section(x)])

        self.n += 1

        old_mu = self.mu
        old_sigma = self.sigma
        n = self.n

        # Incrementally update average and standard deviation
        self.mu = old_mu + (x - old_mu)/n
        self.sigma = np.sqrt(((n-2)/(n-1)) * np.power(old_sigma, 2) + np.power((x - old_mu), 2)/n)

    # TODO: Base prediction on the bins
    def predict(self):
        return np.random.normal(loc=self.mu, scale=self.sigma)

    def get_section(self, x):
        sect_nr = 0
        while sect_nr < self.n_sect:
            boundary = self.mu - self.sect_size * ((self.n_sect - 2) / 2 - sect_nr)
            if x < boundary:
                return sect_nr
            sect_nr += 1
        return sect_nr - 1

    def get_boundaries(self):
        return [self.mu - self.sect_size * ((self.n_sect - 2) / 2 - sect_nr) for sect_nr in range(self.n_sect)]

    def get_counts(self):
        return dict(self.sect_counts)

    def get_probs(self):
        return {k: v / sum(self.sect_counts.values()) for k, v in self.sect_counts.items()}

    def get_params(self):
        if self.parent:
            return [(self.mu, self.sigma)]+ self.parent.get_params()
        return [self.mu, self.sigma]
