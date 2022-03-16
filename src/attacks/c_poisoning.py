from secml.ml.classifiers import CClassifier


class CPoisoning:
    def __init__(self, clf: CClassifier, random_state: int = 999):
        self.clf = clf
        self.random_state = random_state
