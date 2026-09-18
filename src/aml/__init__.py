# On Windows, scikit-learn's bundled OpenMP runtime conflicts with
# TensorFlow's bundled MKL/oneDNN runtime if sklearn loads first in the
# process ("DLL load failed while importing _pywrap_tensorflow_internal").
# Importing tensorflow first, before anything else in this package can pull
# in sklearn, avoids it.
import tensorflow as _tf  # noqa: F401
