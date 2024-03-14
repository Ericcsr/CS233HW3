from setuptools import setup

setup(name='cs233_gtda_hw4',
      version='0.3',
      description='Fourth assignment for CS-233, Stanford',
      author='Panos Achlioptas for Geometric Computing Lab @Stanford',
      author_email='pachlioptas@gmail.com',
      license='MIT',
      install_requires=['torch',
                        'numpy',
                        'scikit-learn',
                        'matplotlib',
                        'tqdm'],
      packages=['cs233_gtda_hw4'],
      zip_safe=False)
