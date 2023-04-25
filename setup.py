from setuptools import setup, find_packages

setup(name='mmpyreader',
      version='0.3',
      description='Simple code to read quickly micro-manager hyperstacks',
      author='Ilyas Djafer-Cherif',
      license='MIT',
      packages=find_packages(),
      install_requires=['pycromanager<=0.16.4'],
      zip_safe=False)
