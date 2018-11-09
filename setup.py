from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='deep',

    version='0.0.1',

    packages=find_packages(exclude=['tests', 'math']),

    package_data = {
    },
    
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'keras',
        'tensorflow',
        'opencv-contrib-python',
        'scikit-learn',
    ],

    tests_require=[
        'tox',
    ],

    setup_requires = [
        'setuptools'
    ],

    include_package_data = True,

    entry_points={
        'console_scripts': [
            'mnist = scripts.mnist:main',
        ],
        'gui_scripts': [
        ],
    },

    license = 'Apache',

    zip_safe=True,

    author='Brendan Drew',

    author_email='',

    description = '',

    long_description=readme(),

    classifiers = [
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Environment :: Console :: Curses',
        'Environment :: X11 Applications :: GTK',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3 :: Only',
        'Topic:: Multimedia:: Video:: Capture',
        'Topic :: Multimedia :: Video :: Display',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Software Development :: Libraries',
        'Topic :: System :: Hardware'
    ],

    keywords = '',

    project_urls = {
        'Issues': '',
        'Source': '',
    },

    url = ''
)
