from setuptools import setup

setup(
    name = 'mmtester',
    packages = ['mmtester'],
    package_data = {'mmtester': ['tester.cfg']},
    version = __import__('mmtester').__version__,
    license = 'MIT',
    description = "Simple local tester for Topcoder's Marathon contests.",
    author = 'Psyho',
    url = 'https://github.com/FakePsyho/mmtester',
    keywords = ['Topcoder', 'Marathon', 'Tester'],
    install_requires=['tabulate'],
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'Environment :: Console',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
    entry_points = {'console_scripts': ['mmtester = mmtester.tester:_main']},
)