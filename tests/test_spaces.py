from pytest import raises

import chromathicity.spaces as sp
from chromathicity.error import UndefinedColorSpaceError


def test_get_space():
    assert sp.get_space('xyz') == ('xyz', sp.XyzData)
    assert sp.get_space('XyzData') == ('xyz', sp.XyzData)
    assert sp.get_space(sp.XyzData) == ('xyz', sp.XyzData)
    with raises(UndefinedColorSpaceError):
        sp.get_space('peanut')
    with raises(UndefinedColorSpaceError):
        sp.get_space(sp.WhitePointSensitive)




