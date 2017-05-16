from abc import ABC, abstractmethod

import numpy as np

from chromathicity.illuminant import Illuminant, get_default_illuminant, D
from chromathicity.error import raise_not_implemented
from chromathicity.util import SetGet
from chromathicity.observer import Observer, get_default_observer, Standard


class RgbSpecification(ABC):
    """
    Specifies the RGB Working space
    
    The concept of colorimetrically defined RGB spaces has been around for a long
    time — at least since the days of the development of color television. But the
    popularity in digital imaging applications grew substantially only after Adobe
    introduced the "RGB Working Space" into Photoshop 5.0. Since that time, many
    different working space definitions have been added to the original set of
    color television spaces.
    """

    def __init__(self):
        self.compander = None

    @property
    def name(self):
        return self.__name__

    @name.setter
    def name(self, n):
        raise_not_implemented(self, 'Setting name')

    @property
    def white_point(self):
        return self.illuminant.get_white_point(self.observer)

    @property
    @abstractmethod
    def illuminant(self) -> Illuminant:
        pass

    @illuminant.setter
    def illuminant(self, ill):
        raise_not_implemented(self, 'Setting illuminant')

    @property
    @abstractmethod
    def observer(self) -> Observer:
        pass

    @observer.setter
    def observer(self, obs):
        raise_not_implemented(self, 'Setting setter')

    @property
    @abstractmethod
    def xyy(self) -> np.ndarray:
        pass

    @xyy.setter
    def xyy(self, x):
        raise_not_implemented(self, 'Setting xyy')

    def compand(self, linear_rgb):
        return self.compander.compand(linear_rgb)

    def inverse_compand(self, companded_rgb):
        return self.compander.inverse_compand(companded_rgb)

    @property
    def linear_transformation(self):
        """
        Compute the linear transformation matrix to convert between RGB and XYZ
        
        see http://brucelindbloom.com/index.html?Eqn_XYZ_to_RGB.html for details 
        about the computation. Note that this transformation matrix is the 
        transpose of the one on the website
        :return: 
        """
        # import the convert function here to avoid circular imports
        from chromathicity.convert import convert
        wp = self.white_point
        xyz = convert(self.xyy, 'xyY', 'CIEXYZ', axis=1)
        xyz_normalized = xyz / xyz[:, 1:2]
        s = np.linalg.solve(xyz_normalized.T, wp[:, np.newaxis])
        return s * xyz_normalized


class Custom(RgbSpecification, SetGet):

    def __init__(self, **kwargs):
        super().__init__()
        self._name = ""
        self._illuminant = get_default_illuminant()
        self._observer = get_default_observer()
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


class Compander(ABC):

    @abstractmethod
    def compand(self, linear_rgb: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_compand(self, companded_rgb: np.ndarray) -> np.ndarray:
        pass


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


def get_default_rgb_specification():
    return Srgb()
