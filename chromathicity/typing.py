from typing import Type, Callable, Any, Optional

from numpy import ndarray

from chromathicity.chromadapt import ChromaticAdaptationAlgorithm
from chromathicity.illuminant import Illuminant
from chromathicity.observer import Observer
from chromathicity.rgbspec import RgbSpecification

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

