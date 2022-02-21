"""
This is a custom implementation of a dataset capable managing R.object files
"""
from pathlib import PurePosixPath
from typing import Any, Dict, List

from kedro.io.core import (
    AbstractVersionedDataSet,
    get_filepath_str,
    get_protocol_and_path,
)

import fsspec
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri

class RDataSet(AbstractVersionedDataSet):
    """``Rdataset`` loads / save R.object data from a given filepath as pandas array.

    Example:
    ::

        >>> ImageDataSet(filepath='/img/file/path.png')
    """

    def __init__(self, filepath: str):
        """Creates a new instance of ImageDataSet to load / save image data at the given filepath.

        Args:
            filepath: The location of the image file to load / save data.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

    def _load(self) -> pd.DataFrame:
        """Loads data from the image file.

        Returns:
            Data from the image file as a numpy array.
        """
        load_path = get_filepath_str(self._get_load_path(), self._protocol)
        with self._fs.open(load_path, mode="r") as f:
            robject = robjects.r['load'](f)
            pandas2ri.activate()
            return robject

    def _save(self) -> None:
        """Saves image data to the specified filepath"""
        print('did nothing')

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset"""
        return dict(filepath=self._filepath, protocol=self._protocol)
