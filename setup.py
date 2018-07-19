from setuptools import setup

setup(
   name='Microbiome',
   version='1.0',
   description='Microbiome',
   author='Microbiome Lab',
   author_email='foomail@foo.com',
   packages=['Microbiome'],  #same as name
   install_requires=['numpy', 'pandas', 'sklearn', 'pytorch'], #external packages as dependencies
)