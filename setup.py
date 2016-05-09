from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'estim n island app',
  ext_modules = cythonize("nisland_model.pyx"),
)