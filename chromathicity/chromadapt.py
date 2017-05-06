from abc import ABC, abstractmethod

import numpy as np


class ChromaticAdaptationAlgorithm(ABC):
    """
    Creates the linear transformations necessary when converting XYZ values 
    between white points. 
    """

    def get_linear_transformation(self, white_point_from, white_point_to):
        m_a = self.cone_response_domain
        cone_response_from = white_point_from.reshape((1, 3)).dot(m_a).reshape(-1)
        cone_response_to = white_point_to.reshape((1, 3)).dot(m_a).reshape(-1)
        response_ratio = np.diag(cone_response_to / cone_response_from)
        return np.linalg.solve(m_a.T, m_a.dot(response_ratio).T).T

    @property
    @abstractmethod
    def cone_response_domain(self) -> np.ndarray:
        pass


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


def get_default_chromatic_adaptation_algorithm():
    return Bradford()
