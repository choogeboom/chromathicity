"""
:mod: chromathicity.spaces -- Color space objects
===================================
.. module:: chromathicity.spaces
   
"""


from abc import ABC, abstractmethod
from typing import Union, Iterable, Tuple, Any

import numpy as np

from chromathicity.chromadapt import (
    get_default_chromatic_adaptation_algorithm,
    ChromaticAdaptationAlgorithm)
from chromathicity.convert import convert
from chromathicity.illuminant import get_default_illuminant, Illuminant
from chromathicity.manage import get_space, get_space_name, color_space
from chromathicity.observer import get_default_observer, Observer
from chromathicity.rgbspec import (get_default_rgb_specification,
                                   RgbSpecification)
from chromathicity.typing import ArrayLike
from chromathicity.util import SetGet, construct_component_inds, \
    get_matching_axis, lazy_property


class ColorSpaceData(ABC):
    """
    Defines the main interface for all color space data classes. 
    :class:`ColorSpaceDataImpl` provides a full implementation of this 
    interface, so all color space data classes should extend that class instead
    of this one.
    """
    __spacename__ = ''

    @abstractmethod
    def get_data(self) -> np.ndarray:
        """
        The stored color data

        Indexing into the space data instance is equivalent to indexing into the
        data itself::

            space.data[inds]

        is the same as::

            space[inds]"""
        pass

    data = lazy_property(get_data)

    def reset_data_cache(self):
        """
        Resets the cached property :attr:`data`
        :return: None
        """
        pass

    @abstractmethod
    def get_axis(self) -> int:
        """
        The axis in the data array that the color components lie along.
        
        Changing the axis will permute the dimensions of the underlying
        array, so that the color space components lie along the new axis
        """
        pass

    @abstractmethod
    def set_axis(self, a: int):
        pass

    axis = lazy_property(get_axis, set_axis)

    @abstractmethod
    def get_illuminant(self) -> Illuminant:
        """
        The illuminant. This combined with the 
        :py:attr:`~ColorSpaceData.observer` 
        determines the reference white point of the space."""
        pass

    @abstractmethod
    def set_illuminant(self, ill: Illuminant):
        pass

    illuminant = lazy_property(get_illuminant, set_illuminant)

    @abstractmethod
    def get_observer(self) -> Observer:
        """
        The observer. This combined with the 
        :py:attr:`~ColorSpaceData.illuminant` 
        determines the reference white point of the space.
        """
        pass

    @abstractmethod
    def set_observer(self, obs: Observer):
        pass

    observer = lazy_property(get_observer, set_observer)

    @abstractmethod
    def get_rgbs(self) -> RgbSpecification:
        """
        The RGB color space specification.
        """
        pass

    @abstractmethod
    def set_rgbs(self, r: RgbSpecification):
        pass

    rgbs = lazy_property(get_rgbs, set_rgbs)

    @abstractmethod
    def get_caa(self) -> ChromaticAdaptationAlgorithm:
        """
        The chromatic adaptation algorithm.
        """
        pass

    @abstractmethod
    def set_caa(self, c: ChromaticAdaptationAlgorithm):
        pass

    caa = lazy_property(get_caa, set_caa)

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

    is_scaled = lazy_property(get_is_scaled, set_is_scaled)

    def get_components(self) -> Tuple[np.ndarray, ...]:
        """
        Tuple containing the correct slices of the data to get the 
        individual color space components. For example, in :class:`LabData`, 
        this property would contain ``(L*, a*, b*)``.

            >>> lab = LabData([[50., 25., 25.], [75., 0., 60.]])
            >>> lab.components[0]
            array([[ 50.],
                   [ 75.]])

        """
        component_inds = construct_component_inds(self.axis,
                                                  self.data.ndim,
                                                  self.num_components,
                                                  min_ndims=0)
        return tuple(self[c] for c in component_inds)

    components = lazy_property(get_components)

    def get_num_components(self) -> int:
        """
        :return: The number of components in the color space. For example 
           :class:`LabData` has three components: L*, a*, b*. 
        """
        return len(self.components)

    num_components = lazy_property(get_num_components)

    @abstractmethod
    def to(self, space: Union[str, type]):
        """
        Convert this space to another space.::
        
            lab = LabData([50., 25., 25.])
            xyz = lab.to('xyz')
        
        :param space: either the name or the class of the destination color 
           space. 
        :return: The new color space data

        """
        pass

    def get_kwargs(self) -> dict:
        """
        The keyword arguments used to construct the object
        :return: 
        """
        return {'axis': self.axis,
                'illuminant': self.illuminant,
                'observer': self.observer,
                'rgbs': self.rgbs,
                'caa': self.caa,
                'is_scaled': self.is_scaled}

    kwargs = property(get_kwargs)

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


class ColorSpaceDataImpl(ColorSpaceData, SetGet):
    """
    A full implementation of the :class:`ColorSpaceDataBase` interface. All 
    color space data classes should extend this.
    
    Since this class is designed to be extended, every property calls a 
    respective getter and setter method, so that overriding is much easier
    in subclasses.
    """

    # Controls how scaling works. If :attr:`~ColorSpaceDataImpl.is_scaled` is
    # ``True``, then the data will be scaled by this value
    scale_factor = 1.  # type: ArrayLike

    # The minimum allowed value for a color space. Data will be clipped to be
    # no less than the value of ``min_value``.
    min_value = [-np.inf]  # type: ArrayLike

    # The maximum allowed value for a color space. Data will be clipped to be
    # no more than the value of ``max_value``.
    max_value = [np.inf]  # type: ArrayLike

    def __init__(self,
                 data: Union[np.ndarray, Iterable[float]],
                 *,
                 axis: int=None,
                 illuminant: Illuminant=get_default_illuminant(),
                 observer: Observer=get_default_observer(),
                 rgbs: RgbSpecification=get_default_rgb_specification(),
                 caa: ChromaticAdaptationAlgorithm=
                 get_default_chromatic_adaptation_algorithm(),
                 is_scaled: bool=False):
        """
        
        :param data: the color space data to contain
        :param axis: the axis along which the color data lies. If `axis` is not
           specified, then it will be determined automatically by finding the 
           last dimension with the required size.
        :param illuminant: the illuminant
        :param observer: the observer
        :param rgbs: the rgb specification
        :param caa: the chromatic adaptation algorithm
        :param is_scaled: Whether or not the data is scaled
        """
        if is_scaled:
            self._data = np.array(data, copy=True)/self.scale_factor
        else:
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
        self._is_scaled = is_scaled

    def get_data(self) -> np.ndarray:
        b_shape = [-1 if k == self.axis else 1
                   for k in range(self._data.ndim)]
        min_value = np.array(self.min_value).reshape(b_shape)
        max_value = np.array(self.max_value).reshape(b_shape)
        d = np.clip(self._data, min_value, max_value)
        return self.scale_factor * d if self.is_scaled else d

    def get_axis(self) -> int:
        return self._axis

    def set_axis(self, a: int):
        if a is None:
            self._axis = get_matching_axis(self.data.shape,
                                           self.num_components)
        elif a != self.axis:
            new_dims = list(range(self.data.ndim))
            new_dims[a] = self.axis
            new_dims[self.axis] = a
            self._data = self._data.transpose(new_dims)
            self._axis = a

    def get_illuminant(self) -> Illuminant:
        return self._illuminant

    def set_illuminant(self, ill: Illuminant):
        self._illuminant = ill

    def get_observer(self) -> Observer:
        return self._observer

    def set_observer(self, obs: Observer):
        self._observer = obs

    def get_rgbs(self) -> RgbSpecification:
        return self._rgbs

    def set_rgbs(self, r: RgbSpecification):
        self._rgbs = r

    def get_caa(self) -> ChromaticAdaptationAlgorithm:
        return self._caa

    def set_caa(self, c: ChromaticAdaptationAlgorithm):
        self._caa = c

    def get_is_scaled(self) -> bool:
        return self._is_scaled

    def set_is_scaled(self, is_scaled: bool) -> None:
        self._is_scaled = is_scaled
        self.reset_data_cache()

    def to(self, space: Union[str, type],
           **kwargs) -> ColorSpaceData:
        to_space, to_class = get_space(space)
        from_space = get_space_name(type(self))
        self_kwargs = self.kwargs
        self_is_scaled = self_kwargs.pop('is_scaled')
        converted_data = convert(self._data,
                                 from_space=from_space,
                                 to_space=to_space,
                                 **self_kwargs)
        new_data = to_class(data=converted_data,
                            **self_kwargs)
        new_data.is_scaled = self_is_scaled
        new_data.set(**kwargs)
        return new_data


@color_space('Spectrum')
class SpectralData(ColorSpaceDataImpl):
    """
    Contains raw reflectance spectral data
    
    """
    scale_factor = 100

    def __init__(self,
                 data: Union[np.ndarray, Iterable[Any], ColorSpaceData],
                 wavelengths: Union[np.ndarray, Iterable[float]],
                 *,
                 axis=None,
                 **kwargs):
        """
        
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
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if axis is None:
            if isinstance(data, ColorSpaceData):
                if data.axis is None:
                    axis = get_matching_axis(data.data.shape, len(wavelengths))
                else:
                    axis = data.axis
            else:
                axis = get_matching_axis(data.shape, len(wavelengths))
        super().__init__(data, axis=axis, **kwargs)
        self._wavelengths = np.array(wavelengths, copy=True)

    def get_num_components(self):
        """The number of wavelengths in the data"""
        return self.wavelengths.size

    def get_wavelengths(self) -> np.ndarray:
        """The wavelengths that correspond to the data"""
        return self._wavelengths

    wavelengths = property(get_wavelengths)

    def get_kwargs(self):
        kwargs = super().get_kwargs()
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

    def set_illuminant(self, ill: Illuminant):
        self.change_white_point(ill, self._observer)

    def set_observer(self, obs):
        self.change_white_point(self._illuminant, obs)

    def change_white_point(self, illuminant: Illuminant, observer: Observer):
        """
        Change the reference white point of the space by setting a new 
        illuminant and observer
        
        :param illuminant: The new illuminant
        :param observer: The new observer
        :return: self
        """
        from chromathicity.convert import xyz2xyz

        source_white_point = self._illuminant.get_white_point(self._observer)
        destination_white_point = illuminant.get_white_point(observer)
        if np.allclose(source_white_point, destination_white_point):
            self._illuminant = illuminant
            self._observer = observer
            return self

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
        self.reset_data_cache()
        return self


@color_space('CIEXYZ')
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
    scale_factor = 100


@color_space('xyY')
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


@color_space('XYZ_r')
class NormalizedXyzData(WhitePointSensitive):
    """
    This space is the CIE XYZ space, normalized by the white point
    """
    pass


@color_space('CIELAB')
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
    min_value = np.array([0., -np.inf, -np.inf])
    max_value = np.array([100., np.inf, np.inf])


@color_space('CIELCH')
class LchData(WhitePointSensitive):
    """
    Represents the CIELCh color space.
     
    From `Wikipedia <https://en.wikipedia.org/wiki/Lab_color_space#
    Cylindrical_representation:_CIELCh_or_CIEHLC>`_:
    
        The CIELCh color space is a CIELab cube color space, where instead of 
        Cartesian coordinates a*, b*, the cylindrical coordinates C* (chroma, 
        relative saturation) and h° (hue angle, angle of the hue in the 
        CIELab color wheel) are specified. The CIELab lightness L* remains 
        unchanged. 
    """
    min_value = np.array([0., 0., 0.])
    max_value = np.array([100., np.inf, 360.])


# noinspection PyMethodOverriding
class RgbsSensitive(WhitePointSensitive):
    """
    This class represents spaces that are sensitive to the choice of the 
    :class:`RgbSpecification`, and will adapt the color data if the 
    :attr:`~RgbsSensitive.rgbs` property value changes.
     
    Any Color space representing RGB data should extend this class.
    """
    scale_factor = 255.
    min_value = np.array([0., 0., 0.])
    max_value = np.array([1., 1., 1.])

    def set_rgbs(self, r: RgbSpecification):
        self.change_rgbs(r, self._caa)

    def set_caa(self, c):
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


@color_space('lRGB')
class LinearRgbData(RgbsSensitive):
    """
    Represents data that is uncompanded RGB
    """
    pass


@color_space('RGB')
class RgbData(RgbsSensitive):
    pass


@color_space('HSL')
class HslData(RgbsSensitive):
    pass


@color_space('HSI')
class HsiData(RgbsSensitive):
    pass


@color_space('HCY')
class HcyData(RgbsSensitive):
    pass
