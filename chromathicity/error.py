
def raise_not_implemented(obj, message: str):
    """
    Raises the ``NotImplementedError`` for convenience
    
    :param obj: 
    :param message: 
    :return: 
    """
    raise NotImplementedError(
        f'{message} is not implemented for objects '
        f'of class {type(obj).__name__}.'
    )
