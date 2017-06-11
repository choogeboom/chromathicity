import numpy as np

from chromathicity.interfaces import ChromaticAdaptationAlgorithm


class Bradford(ChromaticAdaptationAlgorithm):
    """
    This is the current best known chromatic adaptation algorithm
    """

    @property
    def cone_response_domain(self) -> np.ndarray:
        return np.array(
            [
                [0.8951, -0.7502, 0.0389],
                [0.2664, 1.7135, -0.0685],
                [-0.1614, 0.0367, 1.0296]
            ])


class XyzScaling(ChromaticAdaptationAlgorithm):
    """
    This is the "ideal" chromatic adaptation algorithm, but it is the worst.
    """

    @property
    def cone_response_domain(self) -> np.ndarray:
        return np.eye(3)


class VonKries(ChromaticAdaptationAlgorithm):
    """
    Better than XYZ scaling, but worse than Bradford
    """

    @property
    def cone_response_domain(self) -> np.ndarray:
        return np.array(
            [
                [0.40024, -0.2263, 0.0],
                [0.7076, 1.16532, 0.0],
                [-0.08081, 0.0457, 0.91822]
            ])


