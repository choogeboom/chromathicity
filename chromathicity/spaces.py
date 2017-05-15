"""
:mod: chromathicity.spaces -- Color space objects
===================================
.. module:: chromathicity.spaces
   
"""


from abc import ABC, abstractmethod
from copy import copy
import sys
from typing import Union, Iterable, Tuple

from bidict import bidict
import numpy as np

from chromathicity.convert import xyz2xyz, convert, get_matching_axis, \
    construct_component_inds
from chromathicity.chromadapt import (
    get_default_chromatic_adaptation_algorithm, ChromaticAdaptationAlgorithm)
from chromathicity.error import raise_not_implemented, UndefinedColorSpaceError
from chromathicity.illuminant import get_default_illuminant, Illuminant
from chromathicity.mixin import SetGet
from chromathicity.rgbspec import (get_default_rgb_specification,
                                   RgbSpecification)
from chromathicity.observer import get_default_observer, Observer


__all__ = ['get_space', 'get_space_class', 'get_space_name']


# Stores all named color spaces
_space_name_to_type_map = {}


def get_space(space: Union[str, type]):
    """
    Get the space name and class associated with it
    """
    if isinstance(space, str):
        if space in _space_name_to_type_map:
            space_class = _space_name_to_type_map[space]
        else:
            raise UndefinedColorSpaceError(space)
    elif isinstance(space, type) and issubclass(space, ColorSpaceData) \
            and space.__spacename__:
        space_class = space
    else:
        raise TypeError(f'Illegal color space type: {type(space).__name__}')
    space_name = space_class.__spacename__
    return space_name, space_class


def get_space_class(space_name: str):
    """Get the color space class associated with a color space"""
    if isinstance(space_name, str):
        if space_name in _space_name_to_type_map:
            return _space_name_to_type_map[space_name]
        else:
            raise UndefinedColorSpaceError(space_name)
    else:
        raise TypeError('get_space_class expected a str object, but got a '
                        f'{type(space_name).__name__} instead.')


def get_space_name(space_class: type):
    """Get the color space name associated with a color space"""
    if isinstance(space_class, type):
        if issubclass(space_class, ColorSpaceData) \
                and space_class.__spacename__:
            return space_class.__spacename__
        else:
            raise UndefinedColorSpaceError(space_class)
    else:
        raise TypeError('get_space_name expected a type object, but got a '
                        f'{type(space_class).__name__} instead.')


def color_space(*args):
    """
    Decorator that registers a class as a color space. Any number of aliases 
    can be passed.
    
    :return: decorator that returns the class after registering it
    
    Example
    -------
    
    The ``color_space`` decorator registers a class as a color space for color 
    conversions.
    
    .. code-block:: python
       
       @color_space('test1', 'test2')
       class TestSpaceData(ColorSpaceDataImpl):
           pass
       
   
    """

    def decorator(cls: type):
        if args:
            cls.__spacename__ = args[0].lower()
            for name in args:
                _space_name_to_type_map[name.lower()] = cls
        return cls
    return decorator


class ColorSpaceData(ABC):
    """
    Defines the main interface for all color space data classes. 
    :class:`ColorSpaceDataImpl` provides a minimal implementation of this 
    interface, so all color space data classes should extend that class instead
    of this one.
    """
    __spacename__ = ''

    @property
    @abstractmethod
    def data(self):
        pass

    @property
    @abstractmethod
    def axis(self):
        pass

    @axis.setter
    def axis(self, a):
        raise_not_implemented(self, 'setting axis')

    @property
    @abstractmethod
    def illuminant(self):
        pass

    @illuminant.setter
    def illuminant(self, ill: Illuminant):
        raise_not_implemented(self, 'setting illuminant')

    @property
    @abstractmethod
    def observer(self):
        pass

    @observer.setter
    def observer(self, obs: Observer):
        raise_not_implemented(self, 'setting observer')

    @property
    @abstractmethod
    def rgbs(self):
        pass

    @rgbs.setter
    def rgbs(self, r: RgbSpecification):
        raise_not_implemented(self, 'setting rgbs')

    @property
    @abstractmethod
    def caa(self):
        pass

    @caa.setter
    def caa(self, c):
        raise_not_implemented(self, 'setting caa')

    @abstractmethod
    def to(self, space: Union[str, type]):
        pass


class ColorSpaceDataImpl(ColorSpaceData, SetGet):
    """
    A full implementation of the :class:`ColorSpaceDataBase` interface. All 
    color space data classes should extend this.
    """
    def __init__(self,
                 data: Union[np.ndarray, Iterable[float], ColorSpaceData],
                 axis: int=None,
                 illuminant: Illuminant=get_default_illuminant(),
                 observer: Observer=get_default_observer(),
                 rgbs: RgbSpecification=get_default_rgb_specification(),
                 caa: ChromaticAdaptationAlgorithm=get_default_chromatic_adaptation_algorithm()):
        """
        
        :param data: the color space data to contain
        :param axis: the axis along which the color data lies. If `axis` is not
           specified, then it will be determined automatically by finding the 
           last dimension with the required size.
        :param illuminant: the illuminant
        :param observer: the observer
        :param rgbs: the rgb specification
        :param caa: the chromatic adaptation algorithm
        """
        if isinstance(data, ColorSpaceData):
            self._axis = data.axis
            self._illuminant = data.illuminant
            self._observer = data.observer
            self._rgbs = data.rgbs
            self._caa = data.caa
            data = data.data

        self._data = np.array(data, copy=True)
        self._data.flags.writeable = False
        self._axis = (axis
                      if axis is not None
                      else get_matching_axis(self._data.shape, 3))
        self._illuminant = (illuminant
                            if illuminant is not None
                            else get_default_illuminant())
        self._observer = (observer
                          if observer is not None
                          else get_default_observer())
        self._rgbs = (rgbs
                      if rgbs is not None
                      else get_default_rgb_specification())
        self._caa = (caa
                     if caa is not None
                     else get_default_chromatic_adaptation_algorithm())

    def __array__(self, dtype) -> np.ndarray:
        if dtype == self._data.dtype:
            return self._data
        else:
            return np.array(self._data, copy=True, dtype=dtype)

    def __getitem__(self, *args, **kwargs) -> Union[float, np.ndarray]:
        return self.data.__getitem__(*args, **kwargs)

    def __repr__(self) -> str:
        kwarg_repr = ', '.join(f'{key!s}={value!r}'
                               for key, value in self._get_kwargs().items())
        return f'{type(self).__name__}({self.data!r}, {kwarg_repr})'

    def __eq__(self, other) -> bool:
        return (type(self) == type(other)
                and np.allclose(self.data, other.data)
                and self.axis == other.axis
                and self.illuminant == other.illuminant
                and self.observer == other.observer
                and self.rgbs == other.rgbs
                and self.caa == other.caa)

    @property
    def components(self) -> Tuple[np.ndarray, ...]:
        """
        :return: Tuple containing the correct slices of the data to get the 
           individual color space components. For example, in :class:`LabData`, 
           this property would contain ``(L*, a*, b*)``. 
        """
        component_inds = construct_component_inds(self.axis,
                                                  self.data.ndim,
                                                  self.num_components)
        return tuple(self[c] for c in component_inds)

    @property
    def num_components(self) -> int:
        """
        :return: The number of components in the color space. For example 
           :class:`LabData` has three components: L*, a*, b*. 
        """
        return 3

    def to(self, space: Union[str, type],
           **kwargs) -> ColorSpaceData:
        """
        Convert this space to another space.::
        
            lab = LabData([50., 25., 25.])
            xyz = lab.to('xyz')
        
        :param space: either the name or the class of the destination color 
           space. 
        :return: The new color space data

        """
        to_space, to_class = get_space(space)
        from_space = get_space_name(type(self))
        self_kwargs = self._get_kwargs()
        converted_data = convert(self.data,
                                 from_space=from_space,
                                 to_space=to_space,
                                 **self_kwargs)
        new_data = to_class(data=converted_data,
                            **self_kwargs)
        new_data.set(**kwargs)
        return new_data

    def _get_kwargs(self):
        return {'axis': self.axis,
                'illuminant': self.illuminant,
                'observer': self.observer,
                'rgbs': self.rgbs,
                'caa': self.caa}

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def axis(self) -> int:
        return self._axis

    @axis.setter
    def axis(self, a):
        if a is None:
            self.axis = get_matching_axis(self.data.shape,
                                          self.num_components)
        elif a != self.axis:
            new_dims = list(range(self.data.ndim))
            new_dims[a] = self.axis
            new_dims[self.axis] = a
            self._data = self._data.transpose(new_dims)
            self._axis = a

    @property
    def illuminant(self) -> Illuminant:
        return self._illuminant

    @illuminant.setter
    def illuminant(self, ill: Illuminant):
        self._illuminant = ill

    @property
    def observer(self) -> Observer:
        return self._observer

    @observer.setter
    def observer(self, obs: Observer):
        self._observer = obs

    @property
    def rgbs(self) -> RgbSpecification:
        return self._rgbs

    @rgbs.setter
    def rgbs(self, r: RgbSpecification):
        self._rgbs = r

    @property
    def caa(self) -> ChromaticAdaptationAlgorithm:
        return self._caa

    @caa.setter
    def caa(self, c: ChromaticAdaptationAlgorithm):
        self._caa = c


@color_space('spectrum')
class SpectralData(ColorSpaceDataImpl):
    """
    Contains raw reflectance spectral data
    
    In addition to the usual data arguments, this data also needs the 
    wavelengths of the spectra. 
    
    :param data: The spectral data [reflectance].
    :param wavelengths: The wavelengths that correspond to the spectra.
    :param axis: The axis along which the spectra lie.
    :param illuminant: The illuminant
    :type illuminant: Illuminant
    :param observer: The observer
    :type observer: Observer
    :param rgbs: The RGB specification
    :type rgbs: RgbSpecification
    :param caa: The chromatic adaptation algorithm.
    :type caa: ChromaticAdaptationAlgorithm
    """
    def __init__(self, data: Union[np.ndarray, Iterable[float], ColorSpaceData],
                 wavelengths: Union[np.ndarray, Iterable[float]]=None,
                 axis=None,
                 *args,
                 **kwargs):
        if wavelengths is None:
            if isinstance(data, SpectralData) and data.wavelengths is not None:
                wavelengths = data.wavelengths
            else:
                raise TypeError('SpectralData expected wavelength data, but it '
                                'was not specified.')
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if not isinstance(wavelengths, np.ndarray):
            wavelengths = np.array(wavelengths, copy=True)
        if axis is None:
            if isinstance(data, ColorSpaceData):
                if data.axis is None:
                    axis = get_matching_axis(data.data.shape, len(wavelengths))
                else:
                    axis = data.axis
            else:
                axis = get_matching_axis(data.shape, len(wavelengths))
        super().__init__(data, axis, *args, **kwargs)
        self._wavelengths = np.array(wavelengths, copy=True)

    @property
    def num_components(self):
        return self.wavelengths.size

    @property
    def wavelengths(self):
        return self._wavelengths

    def _get_kwargs(self):
        kwargs = super()._get_kwargs()
        kwargs['wavelengths'] = self.wavelengths
        return kwargs


# noinspection PyMethodOverriding
class WhitePointSensitive(ColorSpaceDataImpl):
    """
    This class implements automatic chromatic adaptation whenever the white 
    point of the illuminant/observer combination changes.
    
    Any subclass representing a space that is sensitive to the white point of 
    the illuminant/observer combination should inherit from this class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @ColorSpaceDataImpl.illuminant.setter
    def illuminant(self, ill: Illuminant):
        self.change_white_point(ill, self._observer)

    @ColorSpaceDataImpl.observer.setter
    def observer(self, obs):
        self.change_white_point(self._illuminant, obs)

    def change_white_point(self, illuminant: Illuminant, observer: Observer):
        source_white_point = self._illuminant.get_white_point(self._observer)
        destination_white_point = illuminant.get_white_point(observer)
        if np.allclose(source_white_point, destination_white_point):
            self._illuminant = illuminant
            self._observer = observer
            return

        source_xyz = convert(self._data,
                             from_space=self.__spacename__,
                             to_space='xyz',
                             illuminant=illuminant,
                             observer=observer,
                             rgbs=self._rgbs,
                             caa=self._caa)
        dest_xyz = xyz2xyz(source_xyz,
                           source_white_point=source_white_point,
                           destination_white_point=destination_white_point,
                           axis=self._axis,
                           caa=self._caa)
        self._data = convert(dest_xyz,
                             from_space='xyz',
                             to_space=self.__spacename__,
                             illuminant=illuminant,
                             observer=observer,
                             rgbs=self._rgbs,
                             caa=self._caa)
        self._illuminant = illuminant
        self._observer = observer


@color_space('xyz')
class XyzData(WhitePointSensitive):
    """
    Represents data from the CIE XYZ color space. 
    
    From `Wikipedia <https://en.wikipedia.org/wiki/CIE_1931_color_space>`_:
    
        *The CIE 1931 color spaces were the first defined quantitative links 
        between physical pure colors (i.e. wavelengths) in the 
        electromagnetic visible spectrum, and physiological perceived colors 
        in human color vision. The mathematical relationships that define 
        these color spaces are essential tools for color management, 
        important when dealing with color inks, illuminated displays, 
        and recording devices such as digital cameras.*
    
        *The CIE 1931 RGB color space and CIE 1931 XYZ color space were 
        created by the International Commission on Illumination (CIE) in 
        1931. They resulted from a series of experiments done in the late 
        1920s by William David Wright and John Guild. The experimental 
        results were combined into the specification of the CIE RGB color 
        space, from which the CIE XYZ color space was derived.* 
    """
    pass


@color_space('xyy')
class XyyData(WhitePointSensitive):
    """
    Represents data from the CIE xyY color space
    
    From `Wikipedia <https://en.wikipedia.org/wiki/CIE_1931_color_space#
    CIE_xy_chromaticity_diagram_and_the_CIE_xyY_color_space>`_
    
        Since the human eye has three types of color sensors that respond to 
        different ranges of wavelengths, a full plot of all visible colors is 
        a three-dimensional figure. However, the concept of color can be 
        divided into two parts: brightness and chromaticity. For example, 
        the color white is a bright color, while the color grey is considered 
        to be a less bright version of that same white. In other words, 
        the chromaticity of white and grey are the same while their 
        brightness differs. 

        The CIE XYZ color space was deliberately designed so that the Y 
        parameter is a measure of the luminance of a color. The chromaticity 
        of a color is then specified by the two derived parameters x and y, 
        two of the three normalized values being functions of all three 
        tristimulus values X, Y, and Z.
    """
    pass


@color_space('xyzr')
class NormalizedXyzData(WhitePointSensitive):
    """
    This space is the CIE XYZ space, normalized by the white point
    """
    pass


@color_space('lab')
class LabData(WhitePointSensitive):
    """
    Represents the CIE L*a*b* color space.
    
    From `Wikipedia <https://en.wikipedia.org/wiki/Lab_color_space#CIELAB>`_:
    
        *CIE L\*a\*b\* (CIELAB) is a color space specified by the International 
        Commission on Illumination (French Commission internationale de 
        l'éclairage, hence its CIE initialism). It describes all the colors 
        visible to the human eye and was created to serve as a 
        device-independent model to be used as a reference.* 

        *The three coordinates of CIELAB represent the lightness of the color 
        (L\* = 0 yields black and L\* = 100 indicates diffuse white; specular 
        white may be higher), its position between red/magenta and green (a\*, 
        negative values indicate green while positive values indicate 
        magenta) and its position between yellow and blue (b\*, negative 
        values indicate blue and positive values indicate yellow). The 
        asterisk (\*) after L, a and b are pronounced* star *and are part of the 
        full name, since they represent L\*, a\* and b\*, to distinguish them 
        from Hunter's L, a, and b.*
    """
    pass


@color_space('lchab')
class LchabData(WhitePointSensitive):
    """
    Represents the CIELCh_ab color space.
     
    From `Wikipedia <https://en.wikipedia.org/wiki/Lab_color_space#
    Cylindrical_representation:_CIELCh_or_CIEHLC>`_:
    
        The CIELCh color space is a CIELab cube color space, where instead of 
        Cartesian coordinates a*, b*, the cylindrical coordinates C* (chroma, 
        relative saturation) and h° (hue angle, angle of the hue in the 
        CIELab color wheel) are specified. The CIELab lightness L* remains 
        unchanged. 
    """
    pass


# noinspection PyMethodOverriding
class RgbsSensitive(WhitePointSensitive):
    """
    This class represents spaces that are sensitive to the choice of the 
    `RgbSpecification`, and will adapt the color data if the 
    `rgbs`:instance_attribute property changes 
    """
    @WhitePointSensitive.rgbs.setter
    def rgbs(self, r: RgbSpecification):
        self.change_rgbs(r, self._caa)

    @WhitePointSensitive.caa.setter
    def caa(self, c):
        self.change_rgbs(self._rgbs, c)

    def change_rgbs(self, rgbs, caa):
        xyz = convert(self._data,
                      from_space=self.__spacename__,
                      to_space='xyz',
                      axis=self._axis,
                      illuminant=self._illuminant,
                      observer=self._observer,
                      rgbs=self._rgbs,
                      caa=self._caa)
        self._data = convert(xyz,
                             from_space='xyz',
                             to_space=self.__spacename__,
                             axis=self._axis,
                             illuminant=self._illuminant,
                             observer=self._observer,
                             rgbs=rgbs,
                             caa=caa)
        self._rgbs = rgbs
        self._caa = caa


@color_space('lrgb')
class LinearRgbData(RgbsSensitive):
    """
    Represents data that is uncompanded RGB
    """
    pass


@color_space('rgb')
class RgbData(RgbsSensitive):
    pass


@color_space('hsl')
class HslData(RgbsSensitive):
    pass


@color_space('hsi')
class HsiData(RgbsSensitive):
    pass


@color_space('hcy')
class HcyData(RgbsSensitive):
    pass
