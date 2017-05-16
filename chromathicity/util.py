from typing import Any, Iterable


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
