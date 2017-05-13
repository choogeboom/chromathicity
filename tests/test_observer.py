from pytest import raises
import numpy as np

from chromathicity.observer import Observer, Standard


class TestObserver:

    def test_abstract_observer(self):
        with raises(TypeError):
            Observer()

    def test_standard(self):
        obs = Standard(2)
        assert obs.angle == 2
        assert obs.name == 'CIE Standard 1931 2° Observer'
        assert obs.year == 1931
        assert obs.wavelengths[5] == 365
        assert repr(obs) == "Standard(2)"
