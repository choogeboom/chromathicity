"""
:mod: chromathicity.spaces -- Color space objects
===================================
.. module:: chromathicity.spaces
   
"""


from abc import ABC, abstractmethod
from copy import copy
import sys
from typing import Union, Iterable

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


# Stores all named color spaces
_space_name_map = bidict()


def get_space(space: Union[str, type]):
    """
    Get the space name and class associated with it
    """
    if isinstance(space, str):
        if space in _space_name_map:
            space_name = space
            space_class_name = _space_name_map[space]
        elif space in _space_name_map.inv:
            space_name = _space_name_map.inv[space]
            space_class_name = space
        else:
            raise UndefinedColorSpaceError(space)
        space_class = getattr(sys.modules[__name__], space_class_name)
    elif isinstance(space, type) and issubclass(space, ColorSpaceData):
        if space.__name__ in _space_name_map.inv:
            space_name = _space_name_map.inv[space.__name__]
            space_class = space
        else:
            raise UndefinedColorSpaceError(space)
    else:
        raise TypeError(f'Illegal color space type: {type(space).__name__}')
    return space_name, space_class


def get_space_class(space_name: str):
    """Get the color space class associated with a color space"""
    if space_name in _space_name_map:
        class_name = _space_name_map[space_name]
        return getattr(sys.modules[__name__], class_name)
    else:
        raise UndefinedColorSpaceError(space_name)


def get_space_name(space_type: type):
    """Get the color space name associated with a color space"""
    if space_type.__name__ in _space_name_map.inv:
        return _space_name_map.inv[space_type.__name__]
    else:
        raise UndefinedColorSpaceError(space_type)


def color_space(name: str):
    """
    Decorator that registers a class as a color space
    
    :param name: The name of the color space. This is use to determine which 
                 conversion functions get called when converting to/from the
                 space.
    :return: decorator that returns the class after registering it
    """
    def decorator(cls: type):
        _space_name_map[name] = cls.__name__
        cls.__spacename__ = name
        return cls
    return decorator


class ColorSpaceData(ABC):
    """
    Defines the main interface for all color space data classes
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
    A full implementation of the ColorSpaceDataBase interface. All color space 
    data classes should inherit this.
    """
    def __init__(self,
                 data: Union[np.ndarray, Iterable, ColorSpaceData],
                 axis: int=None,
                 illuminant: Illuminant=None,
                 observer: Observer=None,
                 rgbs: RgbSpecification=None,
                 caa: ChromaticAdaptationAlgorithm=None):
        """
        
        :param data: the color space data to contain
        :param axis: the axis along which the color data lies
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

    def __array__(self, dtype):
        if dtype == self._data.dtype:
            return self._data
        else:
            return np.array(self._data, copy=True, dtype=dtype)

    def __getitem__(self, *args, **kwargs):
        return self.data.__getitem__(*args, **kwargs)

    def __repr__(self):
        kwarg_repr = ', '.join(f'{key!s}={value!r}'
                               for key, value in self._get_kwargs().items())
        return f'{type(self).__name__}({self.data!r}, {kwarg_repr})'

    def __eq__(self, other):
        return (type(self) == type(other)
                and np.allclose(self.data, other.data)
                and self.axis == other.axis
                and self.illuminant == other.illuminant
                and self.observer == other.observer
                and self.rgbs == other.rgbs
                and self.caa == other.caa)

    @property
    def components(self):
        component_inds = construct_component_inds(self.axis,
                                                  self.data.ndim,
                                                  self.num_components)
        return tuple(self[c] for c in component_inds)

    @property
    def num_components(self):
        return 3

    def to(self, space: Union[str, type],
           **kwargs) -> ColorSpaceData:
        """
        Convert this space to another
        
        :param space: either the name of the space, or the class
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
    def data(self):
        return self._data

    @property
    def axis(self):
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
    def illuminant(self):
        return self._illuminant

    @illuminant.setter
    def illuminant(self, ill: Illuminant):
        self._illuminant = ill

    @property
    def observer(self):
        return self._observer

    @observer.setter
    def observer(self, obs: Observer):
        self._observer = obs

    @property
    def rgbs(self):
        return self._rgbs

    @rgbs.setter
    def rgbs(self, r: RgbSpecification):
        self._rgbs = r

    @property
    def caa(self):
        return self._caa

    @caa.setter
    def caa(self, c: ChromaticAdaptationAlgorithm):
        self._caa = c


@color_space('spectrum')
class SpectralData(ColorSpaceDataImpl):
    """
    Contains raw reflectance spectral data
    
    In addition to the usual data arguments, this 
    """
    def __init__(self, data: Union[np.ndarray, Iterable, ColorSpaceData],
                 wavelengths: Union[np.ndarray, Iterable]=None,
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
