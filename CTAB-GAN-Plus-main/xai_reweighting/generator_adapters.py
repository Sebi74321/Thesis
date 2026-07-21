class GeneratorAdapter:
    def fit(self, df):
        raise NotImplementedError

    def sample(self, n):
        raise NotImplementedError