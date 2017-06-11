from abc import ABC, abstractmethod
from typing import Tuple, Union, Type, Iterable, Callable, Any, Optional

import numpy as np
from scipy import integrate as integrate

from chromathicity.error import raise_not_implemented
from chromathicity.math import interp1
from chromathicity.util import construct_component_inds


class Observer(ABC):
    """
    A class to represent standard colorimetric observers

    An observer defines the PSD of the cone response to light, x(l),
    y(l), z(l), where l is the wavelength of light. These are known as the 
    color matching functions, and can be used in conjunction with a specified 
    illuminant to convert between various color spaces.

    There are many pre-defined observers, including a custom observer that 
    allows you to define your own color matching functions. The most common 
    observer used will be the CIE Standard observer.
    """

    def __str__(self):
        return self.name

    @property
    def name(self) -> str:
        return self.__name__

    @name.setter
    def name(self, val):
        raise_not_implemented(self, 'Setting name')

    @property
    @abstractmethod
    def angle(self):
        pass

    @angle.setter
    def angle(self, val):
        raise_not_implemented(self, 'Setting angle')

    @property
    @abstractmethod
    def year(self):
        pass

    @year.setter
    def year(self, val):
        raise_not_implemented(self, 'Setting year')

    @property
    @abstractmethod
    def wavelengths(self) -> np.ndarray:
        pass

    @wavelengths.setter
    def wavelengths(self, val):
        raise_not_implemented(self, 'Setting wavelengths')

    @property
    @abstractmethod
    def xbar(self):
        pass

    @xbar.setter
    def xbar(self, x):
        raise_not_implemented(self, 'Setting xbar')

    @property
    @abstractmethod
    def ybar(self):
        pass

    @ybar.setter
    def ybar(self, y):
        raise_not_implemented(self, 'Setting ybar')

    @property
    @abstractmethod
    def zbar(self) -> np.ndarray:
        pass

    @zbar.setter
    def zbar(self, z):
        raise_not_implemented(self, 'Setting zbar')

    def get_xbar(self, wavelengths: np.ndarray) -> np.ndarray:
        return interp1(wavelengths, self.wavelengths, self.xbar)

    def get_ybar(self, wavelengths: np.ndarray) -> np.ndarray:
        return interp1(wavelengths, self.wavelengths, self.ybar)

    def get_zbar(self, wavelengths: np.ndarray) -> np.ndarray:
        return interp1(wavelengths, self.wavelengths, self.zbar)


class Illuminant(ABC):
    """
    Interface for illuminants

    A standard illuminant is a theoretical source of visible light with a 
    publish profile (power spectral distribution). Standard illuminants
    provide a basis for comparing images or colors recorded under different
    lighting.
    """

    @property
    def name(self) -> str:
        return self.__name__

    def __str__(self):
        return self.name

    @property
    @abstractmethod
    def wavelengths(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def psd(self) -> np.ndarray:
        pass

    def get_psd(self, wavelengths: np.ndarray) -> np.ndarray:
        return interp1(wavelengths, self.wavelengths, self.psd)

    def get_white_point(self, observer: Observer = None) -> np.ndarray:
        """
        Calculate the white point of the illuminant with a specified observer

        :param observer: The observer 
        :return: the white point
        """
        if observer is None:
            from chromathicity.defaults import get_default_observer
            observer = get_default_observer()
        wls = observer.wavelengths
        p = self.get_psd(wls)
        x_power = observer.xbar * p
        is_valid_x = np.logical_not(np.isnan(x_power))
        x_point = integrate.trapz(x_power[is_valid_x], wls[is_valid_x])
        y_power = observer.ybar * p
        is_valid_y = np.logical_not(np.isnan(y_power))
        y_point = integrate.trapz(y_power[is_valid_y], wls[is_valid_y])
        z_power = observer.zbar * p
        is_valid_z = np.logical_not(np.isnan(z_power))
        z_point = integrate.trapz(z_power[is_valid_z], wls[is_valid_z])
        return np.array([x_point, y_point, z_point]) / y_point


class ChromaticAdaptationAlgorithm(ABC):
    """
    Creates the linear transformations necessary when converting XYZ values 
    between white points. 
    """

    def __repr__(self):
        return f'{type(self).__name__}()'

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


class Compander(ABC):

    @abstractmethod
    def compand(self, linear_rgb: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_compand(self, companded_rgb: np.ndarray) -> np.ndarray:
        pass


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
        self.compander: Compander = None

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

    def compand(self, linear_rgb: np.ndarray):
        return self.compander.compand(linear_rgb)

    def inverse_compand(self, companded_rgb: np.ndarray):
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


class ColorSpaceData(ABC):
    """
    Defines the main interface for all color space data classes. 
    :class:`ColorSpaceDataImpl` provides a full implementation of this 
    interface, which makes it much easier to define custom color spaces.
    
    Since this class is designed to be extended, every property calls a 
    respective getter and setter method, so that overriding is much easier
    in subclasses.
    """

    # Controls how scaling works. If :attr:`~ColorSpaceDataImpl.is_scaled` is
    # ``True``, then the data will be scaled by this value
    max_value: np.ndarray = [np.inf]

    # The minimum allowed value for a color space. Data will be clipped to be
    # no less than the value of ``min_value``.
    min_value: np.ndarray = np.array([-np.inf])

    # The maximum allowed value for a color space. Data will be clipped to be
    # no more than the value of ``max_value``.
    scale_factor: np.ndarray = np.array([1.])

    __spacename__: str = ''

    @property
    def data(self) -> np.ndarray:
        """
        The stored color data

        Indexing into the space data instance is equivalent to indexing into 
        the data itself::

            space.data[inds]

        is the same as::

            space[inds]"""
        return self.get_data()

    def get_data(self) -> np.ndarray:
        """
        Return the scaled and clipped data
        """
        b_shape = [-1 if k == self.axis else 1
                   for k in range(self.raw_data.ndim)]
        min_value = np.array(self.min_value).reshape(b_shape)
        max_value = np.array(self.max_value).reshape(b_shape)
        d = np.clip(self.raw_data, min_value, max_value)
        return self.scale_factor * d if self.is_scaled else d

    @property
    def raw_data(self) -> np.ndarray:
        """
        The raw unscaled, un-clipped data. All conversions are done on this
        value.

        :return: the raw data 
        """
        return self.get_raw_data()

    @abstractmethod
    def get_raw_data(self) -> np.ndarray:
        """
        Return the raw unscaled, unclipped color data.
        
        Subclasses should override this method.
        :return: 
        """
        pass

    @property
    def axis(self) -> int:
        """
        The axis in the data array that the color components lie along.
        
        Changing the axis will permute the dimensions of the underlying
        array, so that the color space components lie along the new axis
        """
        return self.get_axis()

    @axis.setter
    def axis(self, a: int):
        self.set_axis(a)

    @abstractmethod
    def get_axis(self) -> int:
        """
        Return the current data axis. Subclasses should override this method.
        :return: 
        """
        pass

    @abstractmethod
    def set_axis(self, a: int):
        pass

    @property
    def illuminant(self) -> Illuminant:
        """
        The illuminant. This combined with the 
        :py:attr:`~ColorSpaceData.observer` 
        determines the reference white point of the space."""
        return self.get_illuminant()

    @illuminant.setter
    def illuminant(self, i: Illuminant):
        self.set_illuminant(i)

    @abstractmethod
    def get_illuminant(self) -> Illuminant:
        pass

    @abstractmethod
    def set_illuminant(self, ill: Illuminant):
        pass

    @property
    def observer(self) -> Observer:
        """
        The observer. This combined with the 
        :py:attr:`~ColorSpaceData.illuminant` 
        determines the reference white point of the space.
        """
        return self.get_observer()

    @observer.setter
    def observer(self, o: Observer):
        self.set_observer(o)

    @abstractmethod
    def get_observer(self) -> Observer:
        pass

    @abstractmethod
    def set_observer(self, obs: Observer):
        pass

    @property
    def rgbs(self) -> RgbSpecification:
        return self.get_rgbs()

    @rgbs.setter
    def rgbs(self, r: RgbSpecification):
        self.set_rgbs(r)

    @abstractmethod
    def get_rgbs(self) -> RgbSpecification:
        """
        The RGB color space specification.
        """
        pass

    @abstractmethod
    def set_rgbs(self, r: RgbSpecification):
        pass

    @property
    def caa(self) -> ChromaticAdaptationAlgorithm:
        return self.get_caa()

    @caa.setter
    def caa(self, c: ChromaticAdaptationAlgorithm):
        self.set_caa(c)

    @abstractmethod
    def get_caa(self) -> ChromaticAdaptationAlgorithm:
        """
        The chromatic adaptation algorithm.
        """
        pass

    @abstractmethod
    def set_caa(self, c: ChromaticAdaptationAlgorithm):
        pass

    @property
    def is_scaled(self) -> bool:
        return self.get_is_scaled()

    @is_scaled.setter
    def is_scaled(self, s: bool):
        self.set_is_scaled(s)

    @abstractmethod
    def get_is_scaled(self) -> bool:
        """
        Whether the data is scaled
        
        If set to ``True``, then the data is scaled by the scale factor of the
        space.
        """
        pass

    @abstractmethod
    def set_is_scaled(self, tf: bool):
        pass

    @property
    def components(self) -> Tuple[np.ndarray, ...]:
        """
        Tuple containing the correct slices of the data to get the 
        individual color space components. For example, in :class:`LabData`, 
        this property would contain ``(L*, a*, b*)``.

            >>> from chromathicity.spaces import LabData
            >>> lab = LabData([[50., 25., 25.], [75., 0., 60.]])
            >>> lab.components[0]
            array([[ 50.],
                   [ 75.]])

        """
        return self.get_components()

    def get_components(self) -> Tuple[np.ndarray, ...]:
        component_inds = construct_component_inds(self.axis,
                                                  self.data.ndim,
                                                  self.num_components,
                                                  min_ndims=0)
        return tuple(self[c] for c in component_inds)

    @property
    def num_components(self) -> int:
        """
        :return: The number of components in the color space. For example 
           :class:`LabData` has three components: L*, a*, b*. 
        """
        return self.get_num_components()

    def get_num_components(self) -> int:
        return 3

    @abstractmethod
    def to(self, space: Union[str, type]) -> 'ColorSpaceData':
        """
        Convert this space to another space.::
        
            >>> from chromathicity.spaces import LabData, XyzData
            >>> lab = LabData([50., 25., 25.])
            >>> xyz = lab.to(XyzData)
        
        :param space: either the name or the class of the destination color 
           space. 
        :return: The new color space data

        """
        pass

    @property
    def kwargs(self) -> dict:
        """
        The keyword arguments used to construct the object
        :return: 
        """
        return self.get_kwargs()

    def get_kwargs(self) -> dict:
        return {'axis': self.axis,
                'illuminant': self.illuminant,
                'observer': self.observer,
                'rgbs': self.rgbs,
                'caa': self.caa,
                'is_scaled': self.is_scaled}

    def __array__(self, dtype) -> np.ndarray:
        if dtype == self.data.dtype:
            return self.data
        else:
            return np.array(self.data, copy=True, dtype=dtype)

    def __repr__(self) -> str:
        kwarg_repr = ', '.join(f'{key!s}={value!r}'
                               for key, value in self.kwargs.items())
        return f'{type(self).__name__}({self.data!r}, {kwarg_repr})'

    def __getitem__(self, *args, **kwargs) -> Union[float, np.ndarray]:
        return self.data.__getitem__(*args, **kwargs)

    def __eq__(self, other) -> bool:
        return (type(self) == type(other)
                and np.allclose(self.data, other.data)
                and self.axis == other.axis
                and self.illuminant == other.illuminant
                and self.observer == other.observer
                and self.rgbs == other.rgbs
                and self.caa == other.caa
                and self.is_scaled == other.is_scaled)


# A type variable for color space types.
ColorSpace = Type[ColorSpaceData]

# A type variable for things that can be turned into arrays.
ArrayLike = Union[np.ndarray, Iterable['ArrayLike'], float]


# A type variable for raw conversions before they are passed to the conversion
# decorator.
BareConversion = Callable[[np.ndarray, Any], np.ndarray]

# After applying the color_conversion decorator, bare conversions will have the
# following signature.
Conversion = Callable[
    [
        np.ndarray,
        Any,
        Optional[int],
        Optional[Illuminant],
        Optional[Observer],
        Optional[RgbSpecification],
        Optional[ChromaticAdaptationAlgorithm]
    ],
    np.ndarray]