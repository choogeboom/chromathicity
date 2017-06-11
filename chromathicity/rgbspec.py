import numpy as np

import chromathicity.defaults
from chromathicity.illuminant import D
from chromathicity.interfaces import RgbSpecification, Compander
from chromathicity.observer import Standard
from chromathicity.util import SetGet


class Custom(RgbSpecification, SetGet):

    def __init__(self, **kwargs):
        super().__init__()
        self._name = ""
        self._illuminant = chromathicity.defaults.get_default_illuminant()
        self._observer = chromathicity.defaults.get_default_observer()
        self._xyy = np.array([[0.6, 0.3, .200],
                              [0.3, 0.6, .800],
                              [0.2, 0.1, .100]])
        self.set(**kwargs)

    def __repr__(self):
        args = ['name', 'illuminant', 'observer', 'xyy']
        kwargs_repr = ', '.join(f'{key}={getattr(self, key)!r}' for key in args)
        return f'Custom({kwargs_repr!s})'

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, n):
        self._name = n

    @property
    def illuminant(self):
        return self._illuminant

    @illuminant.setter
    def illuminant(self, ill):
        self._illuminant = ill

    @property
    def observer(self):
        return self._observer

    @observer.setter
    def observer(self, obs):
        self._observer = obs

    @property
    def xyy(self):
        return self._xyy

    @xyy.setter
    def xyy(self, x):
        self._xyy = x


class Srgb(RgbSpecification):

    def __init__(self):
        super().__init__()
        self.compander = SrgbCompander()

    def __repr__(self):
        return 'Srgb()'

    @property
    def name(self):
        return 'sRGB'

    @property
    def illuminant(self):
        return D('D_65')

    @property
    def observer(self):
        return Standard(2)

    @property
    def xyy(self):
        return np.array([[0.64, 0.33, .212656],
                         [0.30, 0.60, .715158],
                         [0.15, 0.06,  .072186]])


class SrgbCompander(Compander):

    _EPS = 0.0031308
    _DELTA = 12.92
    _ALPHA = 1.055
    _GAMMA = 2.4
    _BETA = 0.055

    def __repr__(self):
        return 'SrgbCompander()'

    def compand(self, linear_rgb: np.ndarray) -> np.ndarray:
        is_small = linear_rgb <= self._EPS
        is_big = np.logical_not(is_small)
        companded_rgb = np.zeros(linear_rgb.shape)
        companded_rgb[is_small] = self._DELTA * linear_rgb[is_small]
        a = self._ALPHA
        g = self._GAMMA
        b = self._BETA
        companded_rgb[is_big] = a*linear_rgb[is_big] ** (1.0/g) - b
        return companded_rgb

    def inverse_compand(self, companded_rgb: np.ndarray) -> np.ndarray:
        is_small = companded_rgb <= self._DELTA*self._EPS
        is_big = np.logical_not(is_small)
        linear_rgb = np.zeros(companded_rgb.shape)
        linear_rgb[is_small] = companded_rgb[is_small] / self._DELTA
        a = self._ALPHA
        g = self._GAMMA
        b = self._BETA
        linear_rgb[is_big] = ((companded_rgb[is_big] + b) / a) ** g
        return linear_rgb


class GammaCompander(Compander):

    def __init__(self, gamma=1):
        self.gamma = gamma

    def __repr__(self):
        return f'GammaCompander({self.gamma!r})'

    def compand(self, linear_rgb: np.ndarray) -> np.ndarray:
        return linear_rgb ** (1.0 / self.gamma)

    def inverse_compand(self, companded_rgb: np.ndarray) -> np.ndarray:
        return companded_rgb ** self.gamma


