import pickle


class Results:
    def __init__(self, file_path):
        self.file_path = file_path


class SingleResult(Results):
    def __init__(self, file_path):
        if not file_path.split(".")[-1] == "single":
            raise Exception("File does not contain single result!")
        super().__init__(file_path)
        res = pickle.load(open(self.file_path, "rb"))
        self.HOUR_LEN = res["HOUR_LEN"]
        self.DAY_LEN = res["DAY_LEN"]
        self.YEAR_LEN = res["YEAR_LEN"]
        self.N_ITER = res["N_ITER"]
        self.LAYER_STATES = res["LAYER_STATES"]
        self.prior = res["prior"]
        self.generated_temps = res["generated_temps"]
        self.predictions = res["predictions"]
        self.agent_params = res["agent_params"]


class StateResults(Results):
    def __init__(self, file_path):
        if not file_path.split(".")[-1] == "states":
            raise Exception("File does not contain state results!")
        super().__init__(file_path)
        res = pickle.load(open(self.file_path, "rb"))
        self.HOUR_LEN = res["HOUR_LEN"]
        self.DAY_LEN = res["DAY_LEN"]
        self.YEAR_LEN = res["YEAR_LEN"]
        self.N_ITER = res["N_ITER"]
        self.LAYER_STATES = res["LAYER_STATES"]
        self.agent_params = res["agent_params"]


class ErrorResults(Results):
    def __init__(self, file_path):
        if not file_path.split(".")[-1] == "errors":
            raise Exception("File does not contain error results!")
        super().__init__(file_path)
        res = pickle.load(open(self.file_path, "rb"))
        self.YEAR_LEN = res["YEAR_LEN"]
        self.N_ITER = res["N_ITER"]
        self.LAYER_STATES = res["LAYER_STATES"]
        self.model_errors = res["model_errors"]