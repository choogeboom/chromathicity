from typing import Union


def raise_not_implemented(obj, message: str):
    """
    Raises the ``NotImplementedError`` for convenience
    
    :param obj: 
    :param message: 
    :return: 
    """
    raise NotImplementedError(
        f'{message} is not implemented for objects '
        f'of type {type(obj).__name__}.'
    )


class ChromathicityError(Exception):
    """ Base class for color exceptions"""
    pass


class UndefinedConversionError(ChromathicityError):
    """ Raised when a conversion is undefined """
    def __init__(self, from_space: str, to_space: str):
        self.message = f'Conversion from {from_space} to ' \
                       f'{to_space} is undefined.'


class UndefinedColorSpaceError(ChromathicityError):
    """ Raised when a color space is not recognized """
    def __init__(self, space: Union[str, type]):
        if isinstance(space, str):
            self.message = f'Color space {space} is unrecognized.'
        elif isinstance(space, type):
            self.message = f'Class {space.__name__} is not a recognized ' \
                           f'color space. Use the color_space decorator.'

