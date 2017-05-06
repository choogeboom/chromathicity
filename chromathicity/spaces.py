from abc import ABC
import sys
from typing import Union

from bidict import bidict
import numpy as np

from chromathicity.convert import xyz2xyz, convert, get_matching_axis
from chromathicity.chromadapt import (
    get_default_chromatic_adaptation_algorithm, ChromaticAdaptationAlgorithm)
from chromathicity.illuminant import get_default_illuminant, Illuminant
from chromathicity.rgbspec import (get_default_rgb_specification,
                                   RgbSpecification)
from chromathicity.observer import get_default_observer, Observer


_space_name_map = bidict()


def color_space(name: str):
    """
    Decorator that adds a class to the _space_name_map
    
    :param name: 
    :return: 
    """
    def decorator(cls: type):
        _space_name_map[name] = cls.__name__
        cls.__spacename__ = name
        return cls
    return decorator


class ColorSpaceDataBase(ABC):

    __spacename__ = ''

    def __init__(self,
                 data: np.ndarray,
                 axis: int=None,
                 illuminant: Illuminant=None,
                 observer: Observer=None,
                 rgbs: RgbSpecification=None,
                 caa: ChromaticAdaptationAlgorithm=None):
        if isinstance(data, ColorSpaceDataBase):
            self._axis = data._axis
            self._illuminant = data._illuminant
            self._observer = data._observer
            self._rgbs = data._rgbs
            self._caa = data._caa
            data = data._data

        self._data = np.array(data, copy=True)
        self._data.flags.writeable = False
        if axis is None:
            self._axis = get_matching_axis(self._data.shape, 3)
        else:
            self._axis = axis
        if illuminant is None:
            self._illuminant = get_default_illuminant()
        else:
            self._illuminant = illuminant
        if observer is None:
            self._observer = get_default_observer()
        else:
            self._observer = observer
        if rgbs is None:
            self._rgbs = get_default_rgb_specification()
        else:
            self._rgbs = rgbs
        if caa is None:
            self._caa = get_default_chromatic_adaptation_algorithm()
        else:
            self._caa = caa

    def __array__(self, dtype):
        if dtype == self._data.dtype:
            return self._data
        else:
            return np.array(self._data, copy=True, dtype=dtype)

    def to(self, space: Union[str, type]):
        if isinstance(space, str):
            if space in _space_name_map:
                to_space = space
                to_class_name = _space_name_map[space]
            elif space in _space_name_map.inv:
                to_space = _space_name_map.inv[space]
                to_class_name = space
            else:
                raise ValueError(f'Illegal color space: {space}')
            to_class = getattr(sys.modules[__name__], to_class_name)
        elif isinstance(space, type):
            if issubclass(space, ColorSpaceDataBase):
                to_space = _space_name_map.inv[space.__name__]
                to_class = space
            else:
                raise ValueError(f'Illegal color space type: {space.__name__}')
        else:
            raise TypeError(f'Illegal color space type: {type(space).__name__}')
        from_space = _space_name_map.inv[type(self).__name__]
        if from_space == to_space:
            return self
        converted_data = convert(self._data,
                                 from_space=from_space,
                                 to_space=to_space,
                                 illuminant=self._illuminant,
                                 observer=self._observer,
                                 rgbs=self._rgbs,
                                 caa=self._caa)
        return self._construct_space(to_class, converted_data)

    def _construct_space(self, space, new_data):
        return space(data=new_data,
                     axis=self._axis,
                     illuminant=self._illuminant,
                     observer=self._observer,
                     rgbs=self._rgbs,
                     caa=self._caa)

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
class SpectralData(ColorSpaceDataBase):

    def __init__(self, data, wavelengths: np.ndarray, axis, *args, **kwargs):
        if axis is None:
            self._axis = get_matching_axis(self._data.shape, wavelengths.size)
        super().__init__(data, axis, *args, **kwargs)
        self._wavelengths = wavelengths

    def change_white_point(self, illuminant: Illuminant, observer: Observer):
        self._illuminant = illuminant
        self._observer = observer


class WhitePointSensitive(ColorSpaceDataBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def illuminant(self):
        return self._illuminant

    @illuminant.setter
    def illuminant(self, ill):
        self.change_white_point(ill, self._observer)

    @property
    def observer(self):
        return self._observer

    @observer.setter
    def observer(self, obs):
        self.change_white_point(self._illuminant, obs)

    def change_white_point(self, illuminant: Illuminant, observer: Observer):
        source_xyz = convert(self._data,
                             from_space=self.__spacename__,
                             to_space='xyz',
                             illuminant=illuminant,
                             observer=observer,
                             rgbs=self._rgbs,
                             caa=self._caa)
        source_white_point = self._illuminant.get_white_point(self._observer)
        destination_white_point = illuminant.get_white_point(observer)
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
    pass


@color_space('xyzr')
class NormalizedXyzData(WhitePointSensitive):
    pass


@color_space('lab')
class LabData(WhitePointSensitive):
    pass


class RgbsSensitive(WhitePointSensitive):

    @property
    def rgbs(self):
        return self._rgbs

    @rgbs.setter
    def rgbs(self, r: RgbSpecification):
        self.change_rgbs(r, self._caa)

    @property
    def caa(self):
        return self._caa

    @caa.setter
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
    pass


@color_space('rgb')
class RgbData(RgbsSensitive):
    pass


@color_space('xyy')
class XyyData(WhitePointSensitive):
    pass



