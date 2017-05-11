from chromathicity.chromadapt import Bradford


def test_bradford():
    b = Bradford()
    assert repr(b) == 'Bradford()'
