"""
Some of the code in this module was taken and modified from the python-colormath 
package, which is located at 

    https://github.com/gtaylor/python-colormath

Below is the Copyright notice for that code:

Copyright (c) 2014, Greg Taylor
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this 
   list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice, 
   this list of conditions and the following disclaimer in the documentation 
   and/or other materials provided with the distribution.
 * Neither the name of the DUO Interactive, LLC nor the names of its 
   contributors may be used to endorse or promote products derived from this 
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE 
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from abc import ABC, abstractmethod
from functools import wraps
from inspect import signature
from logging import getLogger
from typing import Union, Callable, List, Tuple, Dict, Any, Optional, Type, \
    TYPE_CHECKING

import numpy as np
from networkx import DiGraph, shortest_path, NetworkXNoPath

from chromathicity.error import UndefinedConversionError, \
    UndefinedColorSpaceError
from chromathicity.illuminant import Illuminant
from chromathicity.observer import Observer
from chromathicity.rgbspec import RgbSpecification
from chromathicity.chromadapt import ChromaticAdaptationAlgorithm
import chromathicity.spaces

logger = getLogger(__name__)


class ConversionManager(ABC):

    def __init__(self):
        self.registered_color_spaces = set()

    def add_type_conversion(self, start_type, target_type, conversion_function):
        """
        Register a conversion function between two color spaces.
        :param start_type: Starting color space.
        :param target_type: Target color space.
        :param conversion_function: Conversion function.
        """
        self.registered_color_spaces.add(start_type)
        self.registered_color_spaces.add(target_type)
        logger.debug('Registered conversion from %s to %s', start_type,
                     target_type)

    @abstractmethod
    def get_conversion_path(self, start_space_name, target_space_name):
        """
        Return a list of conversion functions that, if applied iteratively on 
        a color of the ``start_space_name`` color space, result in a color in 
        the ``result_space_name`` color space. Raises an 
        :exc:`UndefinedConversionError` if no valid conversion path can be 
        found. 
        
        :param start_space_name: Starting color space type.
        :param target_space_name: Target color space type.
        :return: List of conversion functions.
        """
        pass


class GraphConversionManager(ConversionManager):
    def __init__(self):
        super().__init__()
        self.conversion_graph = DiGraph()

    def get_conversion_path(self, start_space_name, target_space_name):
        if start_space_name not in self.conversion_graph:
            raise UndefinedColorSpaceError(start_space_name)
        if target_space_name not in self.conversion_graph:
            raise UndefinedColorSpaceError(target_space_name)
        try:
            # Retrieve node sequence that leads from start_space_name to
            # target_space_name.
            path = shortest_path(self.conversion_graph, start_space_name,
                                 target_space_name)
        except NetworkXNoPath:
            raise UndefinedConversionError(
                start_space_name,
                target_space_name,
            )
        # Look up edges between nodes and retrieve the conversion function
        # for each edge.
        cf = 'conversion_function'
        return [self.conversion_graph.get_edge_data(node_a, node_b)[cf]
                for node_a, node_b in zip(path[:-1], path[1:])]

    def add_type_conversion(self, start_space_name, target_space_name,
                            conversion_function):
        super().add_type_conversion(start_space_name,
                                    target_space_name,
                                    conversion_function)
        self.conversion_graph.add_edge(
            start_space_name,
            target_space_name,
            {'conversion_function': conversion_function})


class DummyConversionManager(ConversionManager):
    def add_type_conversion(self, start_space_name, target_space_name,
                            conversion_function):
        pass

    def get_conversion_path(self, start_space_name, target_space_name):
        raise UndefinedConversionError(
            start_space_name,
            target_space_name,
        )


_conversion_manager = GraphConversionManager()

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


def color_conversion(from_space_name: str, to_space_name: str) \
        -> Callable[[BareConversion], Conversion]:
    """
    
    Decorator to indicate a function that performs a conversion from one 
    color space to another. This decorator wraps the original function, 
    giving every conversion the same signature:
    
        conversion(data, *args, axis, illuminant, observer, rgbs, caa)
    
    In addition, the function will be registered as a conversion, so it can 
    be used to perform color space transformations between color spaces that 
    do not have direct conversion functions (e.g., Luv to CMYK). The path 
    between two spaces can be found using the :func:`get_conversion_path` 
    function. 
    
    :param from_space_name: Starting color space name or type
    :param to_space_name: Target color space name or type
    """

    def decorator(f: BareConversion) -> Conversion:
        f.start_type = from_space_name
        f.target_type = to_space_name

        f_sig = signature(f)
        kwarg_names = {n for n, p in f_sig.parameters.items()
                       if p.kind == p.KEYWORD_ONLY}
        f._kwarg_names = kwarg_names

        @wraps(f, assigned=('__module__', '__name__', '__qualname__',
                            '__doc__'))
        def convert_wrapper(
                data: np.ndarray,
                *args,
                axis: int=None,
                illuminant: Illuminant=None,
                observer: Observer=None,
                rgbs: RgbSpecification=None,
                caa: ChromaticAdaptationAlgorithm=None) \
                -> np.ndarray:
            local_values = locals()
            kwargs = {n: local_values[n]
                      for n in local_values.keys() & kwarg_names}
            return f(data, *args, **kwargs)

        _conversion_manager.add_type_conversion(from_space_name,
                                                to_space_name,
                                                convert_wrapper)
        return convert_wrapper

    return decorator


def get_conversion_path(from_space: str, to_space: str) -> List[Conversion]:
    """
    Returns a list of functions to apply to perform the conversion. Raises an 
    :class:`UndefinedConversionError` if a conversion path is not found. 
    
    :param from_space: The starting space name
    :param to_space: the destination space name
    """
    return _conversion_manager.get_conversion_path(from_space, to_space)
