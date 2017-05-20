from typing import Type, Union, Iterable, Callable, Any, Optional

from numpy import ndarray
from chromathicity.illuminant import Illuminant
from chromathicity.observer import Observer
from chromathicity.rgbspec import RgbSpecification
from chromathicity.chromadapt import ChromaticAdaptationAlgorithm

ArrayLike = Union['ndarray', float, Iterable['ArrayLike']]

ColorSpace = Type['chromathicity.ColorSpaceData']

Conversion = Callable[
    [
        ndarray,
        Any,
        Optional[int],
        Optional[Illuminant],
        Optional[Observer],
        Optional[RgbSpecification],
        Optional[ChromaticAdaptationAlgorithm]
    ],
    ndarray]

