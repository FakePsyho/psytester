from setuptools import setup

setup(
    name = 'mmtester',
    packages = ['mmtester'],
    package_data = {'mmtester': ['topcoder.cfg', 'old_topcoder.cfg', 'atcoder.cfg']},
    version = __import__('mmtester').__version__,
    license = 'MIT',
    description = "Local tester for Topcoder Marathons & AtCoder Heuristic Contests",
    author = 'Psyho',
    url = 'https://github.com/FakePsyho/mmtester',
    keywords = ['Topcoder', 'Marathon', 'AtCoder', 'Tester'],
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