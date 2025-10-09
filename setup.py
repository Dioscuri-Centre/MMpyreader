from setuptools import setup, find_packages

setup(name='mmpyreader',
      version='0.4.3',
      description='Simple code to read quickly micro-manager hyperstacks',
      author='Ilyas Djafer-Cherif',
      license='MIT',
      packages=find_packages(),
      install_requires=['pycromanager==0.34.8','tqdm'],
      zip_safe=False)
