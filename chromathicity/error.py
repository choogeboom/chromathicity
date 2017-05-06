
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
    pass


