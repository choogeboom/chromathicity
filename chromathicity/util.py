from typing import Any, Iterable, Tuple, TypeVar, Generic, Callable, Union

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

A = TypeVar('A')
T = TypeVar('T')
GetMethod = Callable[..., A]
SetMethod = Callable[[T, A], None]
DelMethod = Callable[[T], None]


# noinspection PyPep8Naming
class lazy_property(Generic[A]):
    """
    A property-like descriptor that does not bind to a function, but to the name 
    of the function. That way subclasses can easily override the getter/setter/
    delete
    """
    def __init__(self,
                 getter_method: GetMethod=None,
                 setter_method: SetMethod=None,
                 deleter_method: DelMethod=None,
                 doc=None):
        self.getter_method = getter_method
        self.setter_method = setter_method
        self.deleter_method = deleter_method
        if doc is None:
            if getter_method.__doc__ is not None:
                doc = getter_method.__doc__
            elif setter_method is not None:
                doc = setter_method.__doc__
        self.__doc__ = doc

    def __get__(self, obj, cls=None) -> Union['lazy_property', A]:
        if obj is None:
            return self
        if self.getter_method is None:
            raise AttributeError('unreadable attribute')
        try:
            fget = getattr(obj, self.getter_method.__name__)
        except AttributeError:
            raise TypeError(f'{type(obj).__name__} object does not have '
                            f'a {self.getter_method.__name__} method')
        return fget()

    def __set__(self, obj, value: A):
        if self.setter_method is None:
            raise AttributeError("can't set attribute")
        try:
            fset = getattr(obj, self.setter_method.__name__)
        except AttributeError:
            raise TypeError(f'{type(obj).__name__} object does not have '
                            f'a {self.setter_method.__name__} method.')
        fset(value)

    def __delete__(self, obj):
        if self.deleter_method is None:
            raise AttributeError("can't delete attribute")
        try:
            fdel = getattr(obj, self.deleter_method.__name__)
        except AttributeError:
            raise TypeError(f'{type(obj).__name__} object does not have '
                            f'a {self.deleter_method.__name__} method')
        fdel()
