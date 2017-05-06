
class SetGet:
    """
    Provides a useful set and get interface
    """

    def set(self, **kwargs):
        """
        Set attributes
        
        :param kwargs: 
        :return: 
        """
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def get(self, *args):
        """
        Get a number of attributes
        
        :param args: 
        :return: 
        """
        return tuple(getattr(self, key) for key in args)
