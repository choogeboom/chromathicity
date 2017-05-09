from pytest import raises

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
        assert obs.xbar[5] == 0.000232100000
        assert obs.ybar[5] == 0.000006965000
        assert obs.zbar[5] == 0.001086000000
        assert repr(obs) == "Standard(2)"
