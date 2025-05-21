try:
    from . import gw_utils
    from . import mnist_dataset
except ImportError:
    import logging
    logging.warning("Some modules could not be imported in __init__.py")
