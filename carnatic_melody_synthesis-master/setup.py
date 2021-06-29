from setuptools import setup

setup(name='carnatic_melody_synthesis',
      description='Generator of predominant melody annotations for Carnatic Music',
      version='1.0',
      author='Genis Plaja',
      author_email='genis DOT plaja01 AT estudiant DOT upf DOT edu',
      license='agpl 3.0',
      url='https://github.com/genisplaja/carnatic_melody_synthesis',
      packages=['carnatic_melody_synthesis'],
      install_requires=[
          "numpy",
          "scipy",
      ],
)