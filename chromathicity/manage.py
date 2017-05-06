"""
Much of the code in this module was taken and modified from the python-colormath 
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
from logging import getLogger

from networkx import DiGraph, shortest_path, NetworkXNoPath

from chromathicity.error import UndefinedConversionError

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
        logger.debug('Registered conversion from %s to %s', start_type, target_type)

    @abstractmethod
    def get_conversion_path(self, start_type, target_type):
        """
        Return a list of conversion functions that if applied iteratively on a color of the start_type color space result
        in a color in the result_type color space.
        Raises an UndefinedConversionError if no valid conversion path can be found.
        :param start_type: Starting color space type.
        :param target_type: Target color space type.
        :return: List of conversion functions.
        """
        pass


class GraphConversionManager(ConversionManager):
    def __init__(self):
        super().__init__()
        self.conversion_graph = DiGraph()

    def get_conversion_path(self, start_type, target_type):
        try:
            # Retrieve node sequence that leads from start_type to target_type.
            path = shortest_path(self.conversion_graph, start_type, target_type)
        except NetworkXNoPath:
            raise UndefinedConversionError(
                start_type,
                target_type,
            )
        # Look up edges between nodes and retrieve the conversion function
        # for each edge.
        cf = 'conversion_function'
        return [self.conversion_graph.get_edge_data(node_a, node_b)[cf]
                for node_a, node_b in zip(path[:-1], path[1:])]

    def add_type_conversion(self, start_type, target_type, conversion_function):
        super().add_type_conversion(start_type,
                                    target_type,
                                    conversion_function)
        self.conversion_graph.add_edge(
            start_type,
            target_type,
            {'conversion_function': conversion_function})


class DummyConversionManager(ConversionManager):
    def add_type_conversion(self, start_type, target_type, conversion_function):
        pass

    def get_conversion_path(self, start_type, target_type):
        raise UndefinedConversionError(
            start_type,
            target_type,
        )


_conversion_manager = GraphConversionManager()


def color_conversion(start_type, target_type):
    """
    
    Decorator to indicate a function that performs a conversion from one 
    color space to another. This decorator will return the original function 
    unmodified, however it will be registered in the _conversion_manager so 
    it can be used to perform color space transformations between color 
    spaces that do not have direct conversion functions (e.g., Luv to CMYK). 
    
    Note: For a conversion to/from RGB supply the BaseRGBColor class.
    
    :param start_type: Starting color space type
    :param target_type: Target color space type
    """

    def decorator(f):
        f.start_type = start_type
        f.target_type = target_type
        _conversion_manager.add_type_conversion(start_type, target_type, f)
        return f

    return decorator


def get_conversion_path(from_space, to_space):
    """ Returns a list of functions to apply to perform the conversion """
    return _conversion_manager.get_conversion_path(from_space, to_space)
