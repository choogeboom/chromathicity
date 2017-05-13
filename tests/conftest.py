from numpy import set_printoptions


def pytest_runtest_setup():
    set_printoptions(precision=10)
