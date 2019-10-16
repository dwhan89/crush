'''source finder doc

currently, crush uses nemo library (https://github.com/simonsobs/nemo) to search for point sources in maps using
real-space matched filter. therefore, this module is designed to handle the interaction between crush and nemo.
having said that, there is necessary for crush to use nemo exclusively. in the future release, we would like to make
this modulus so that we can plug in any alternative source finder libraries.

'''

import numpy as np

try:
    import nemo
except ImportError:
    pass


class SourceFinder(object):
    """
    define the top level api for any SourceFinder class and subclass. any specific implementation of sourcefinder
    classes should inherit from this class.
    """

    def __init__(self):
        pass


class Nemo(SourceFinder):
    """
    this class handles top interactions with nemo library (https://github.com/simonsobs/nemo)
    """

    def __init__(self):
        super().__init__()
        pass
