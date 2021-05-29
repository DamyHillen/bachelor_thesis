import pickle


class Results:
    def __init__(self, file_path):
        self.N_ITER, self.YEAR_LEN, self.LAYER_STATES, self.model_errors = pickle.load(open(file_path, "rb"))

    # def
