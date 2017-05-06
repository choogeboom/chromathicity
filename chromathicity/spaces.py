from abc import ABC
import sys
from typing import Union
from copy import deepcopy

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


class ColorSpaceData(ABC):

    def __init__(self,
                 data: np.ndarray,
                 axis: int=None,
                 illuminant: Illuminant=None,
                 observer: Observer=None,
                 rgbs: RgbSpecification=None,
                 caa: ChromaticAdaptationAlgorithm=None):
        self._data = np.array(data, copy=True)
        if axis is None:
            self._axis = get_matching_axis(self._data.shape, 3)
        else:
            self._axis = axis
        if illuminant is None:
            self._illuminant = get_default_illuminant()
        else:
            self._illuminant = deepcopy(illuminant)
        if observer is None:
            self._observer = get_default_observer()
        else:
            self._observer = deepcopy(observer)
        if rgbs is None:
            self._rgbs = get_default_rgb_specification()
        else:
            self._rgbs = deepcopy(rgbs)
        if caa is None:
            self._caa = get_default_chromatic_adaptation_algorithm()
        else:
            self._caa = deepcopy(caa)

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
            if issubclass(space, ColorSpaceData):
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
                     illuminant=deepcopy(self._illuminant),
                     observer=deepcopy(self._observer),
                     rgbs=deepcopy(self._rgbs),
                     caa=deepcopy(self._caa))


@color_space('xyz')
class XyzData(ColorSpaceData):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@color_space('lab')
class LabData(ColorSpaceData):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


