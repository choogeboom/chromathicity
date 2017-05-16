from typing import Any, Iterable, Tuple

import numpy as np


class SetGet:
    """
    Provides a useful set and get interface
    """

    def set(self, **kwargs):
        """
        Set attributes::
        
            obj.set(attr1='value', attr2=35, attr3=True)
        
        :param kwargs: ``name=value`` pairs of attributes to set
        :return: self
        """
        for key in kwargs:
            setattr(self, key, kwargs[key])
        return self

    def get(self, *args) -> Iterable[Any]:
        """
        Get a number of attributes::
        
            obj.get('attr1', 'attr2', 'attr3')
        
        :param args: a number of attribute names to return
        :return: An iterable containing the attributes
        """
        return (getattr(self, key) for key in args)


def construct_component_inds(axis: int,
                             n_dims: int,
                             n_components: int,
                             min_ndims: int=2) -> Tuple[Tuple]:
    """
    Construct a tuple of tuples, where each element extracts the correct 
    component values.
    
    :param axis: 
    :param n_dims: 
    :param n_components: 
    :param min_ndims:
    :return: 
    """
    # noinspection PyTypeChecker
    return tuple(
        tuple(slice(i, i+1)
              if dim == axis
              else (slice(None) if dim < n_dims else np.newaxis)
              for dim in range(max(n_dims, min_ndims)))
        for i in range(n_components))


def get_matching_axis(shape: Tuple, length: int) -> int:
    """
    Infers the correct axis to use
    
    :param shape: the shape of the input
    :param length: the desired length of the axis
    :return: the correct axis. If multiple axes match, then it returns the last 
             one.
    """
    # noinspection PyUnresolvedReferences
    axis_candidates = np.nonzero(np.array(shape) == length)[0]
    if len(axis_candidates) == 0:
        raise ValueError('Unable to infer axis tue to shape mismatch: ' 
                         '{} =/= {}.'.format(shape, length))
    return axis_candidates[-1]