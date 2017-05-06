from setuptools import setup

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: BSD License',
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Programming Language :: Python :: 3.6',
]

KEYWORDS = 'color color-space cielab ciexyz math conversion convert'

setup(
    name='chromathicity',
    version='0.1.0',
    description='Extensible color conversions',
    long_description='',
    url='https://github.com/hoogamaphone/chromathicity',
    author='Christopher Hoogeboom',
    author_email='chris.hoogeboom@gmail.com',
    license='BSD',
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    packages=['chromathicity'],
    install_requires=['scipy', 'bidict', 'networkx']
)
