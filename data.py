from collections import defaultdict
from functools import reduce
import numbers
import struct

import numpy as np
import spectral.io.envi
from scipy.interpolate import interp1d
from scipy.io import matlab

import Orange
from Orange.data import \
    ContinuousVariable, StringVariable, TimeVariable, Domain, Table
from Orange.data.io import FileFormat
import Orange.data.io

# from .pymca5 import OmnicMap
from agilent import agilentImage, agilentMosaic, agilentImageIFG, agilentMosaicIFG


class SpectralFileFormat:

    def read_spectra(self):
        """ Fast reading of spectra. Return spectral information
        in two arrays (wavelengths and values). Only additional
        attributes (usually metas) are returned as a Table.

        Return a triplet:
            - 1D numpy array,
            - 2D numpy array with the same last dimension as xs,
            - Orange.data.Table with only meta or class attributes
        """
        pass

    def read(self):
        return build_spec_table(*self.read_spectra())


def _spectra_from_image(X, features, x_locs, y_locs):
    """
    Create a spectral format (returned by SpectralFileFormat.read_spectra)
    from 3D image organized [ rows, columns, wavelengths ]
    """
    X = np.asarray(X)
    x_locs = np.asarray(x_locs)
    y_locs = np.asarray(y_locs)

    # each spectrum has its own row
    spectra = X.reshape((X.shape[0]*X.shape[1], X.shape[2]))

    # locations
    y_loc = np.repeat(np.arange(X.shape[0]), X.shape[1])
    x_loc = np.tile(np.arange(X.shape[1]), X.shape[0])
    metas = np.array([x_locs[x_loc], y_locs[y_loc]]).T

    domain = Orange.data.Domain([], None,
                                metas=[Orange.data.ContinuousVariable.make("map_x"),
                                       Orange.data.ContinuousVariable.make("map_y")]
                                )
    data = Orange.data.Table.from_numpy(domain, X=np.zeros((len(spectra), 0)),
                                        metas=np.asarray(metas, dtype=object))
    return features, spectra, data


class AgilentImageReader(FileFormat, SpectralFileFormat):
    """ Reader for Agilent FPA single tile image files"""
    EXTENSIONS = ('.dat',)
    DESCRIPTION = 'Agilent Single Tile Image'

    def __init__(self, path):
        self.filename = path

    def read_spectra(self):
        ai = agilentImage(self.filename)
        info = ai.info
        X = ai.data

        try:
            features = info['wavenumbers']
        except KeyError:
            # just start counting from 0 when nothing is known
            features = np.arange(X.shape[-1])

        try:
            px_size = info['FPA Pixel Size'] * info['PixelAggregationSize']
        except KeyError:
            # Use pixel units if FPA Pixel Size is not known
            px_size = 1
        x_locs = np.linspace(
            0, X.shape[1]*px_size, num=X.shape[1], endpoint=False)
        y_locs = np.linspace(
            0, X.shape[0]*px_size, num=X.shape[0], endpoint=False)

        return _spectra_from_image(X, features, x_locs, y_locs)


class AgilentImageIFGReader(FileFormat, SpectralFileFormat):
    """ Reader for Agilent FPA single tile image files (IFG)"""
    EXTENSIONS = ('.seq',)
    DESCRIPTION = 'Agilent Single Tile Image (IFG)'

    def read_spectra(self):
        ai = agilentImageIFG(self.filename)
        info = ai.info
        X = ai.data

        features = np.arange(X.shape[-1])

        try:
            px_size = info['FPA Pixel Size'] * info['PixelAggregationSize']
        except KeyError:
            # Use pixel units if FPA Pixel Size is not known
            px_size = 1
        x_locs = np.linspace(
            0, X.shape[1]*px_size, num=X.shape[1], endpoint=False)
        y_locs = np.linspace(
            0, X.shape[0]*px_size, num=X.shape[0], endpoint=False)

        features, data, additional_table = _spectra_from_image(
            X, features, x_locs, y_locs)

        import_params = ['Effective Laser Wavenumber',
                         'Under Sampling Ratio',
                         ]
        new_attributes = []
        new_columns = []
        for param_key in import_params:
            try:
                param = info[param_key]
            except KeyError:
                pass
            else:
                new_attributes.append(ContinuousVariable.make(param_key))
                new_columns.append(np.full((len(data),), param))

        domain = Domain(additional_table.domain.attributes,
                        additional_table.domain.class_vars,
                        additional_table.domain.metas + tuple(new_attributes))
        table = additional_table.transform(domain)
        table[:, new_attributes] = np.asarray(new_columns).T

        return (features, data, table)


class agilentMosaicReader(FileFormat, SpectralFileFormat):
    """ Reader for Agilent FPA mosaic image files"""
    EXTENSIONS = ('.dmt',)
    DESCRIPTION = 'Agilent Mosaic Image'

    def read_spectra(self):
        am = agilentMosaic(self.filename)
        info = am.info
        X = am.data

        try:
            features = info['wavenumbers']
        except KeyError:
            # just start counting from 0 when nothing is known
            features = np.arange(X.shape[-1])

        try:
            px_size = info['FPA Pixel Size'] * info['PixelAggregationSize']
        except KeyError:
            # Use pixel units if FPA Pixel Size is not known
            px_size = 1
        x_locs = np.linspace(
            0, X.shape[1]*px_size, num=X.shape[1], endpoint=False)
        y_locs = np.linspace(
            0, X.shape[0]*px_size, num=X.shape[0], endpoint=False)

        return _spectra_from_image(X, features, x_locs, y_locs)


class agilentMosaicIFGReader(FileFormat, SpectralFileFormat):
    """ Reader for Agilent FPA mosaic image files"""
    EXTENSIONS = ('.dmt',)
    DESCRIPTION = 'Agilent Mosaic Image (IFG)'
    PRIORITY = agilentMosaicReader.PRIORITY + 1

    def read_spectra(self):
        am = agilentMosaicIFG(self.filename)
        info = am.info
        X = am.data

        features = np.arange(X.shape[-1])

        try:
            px_size = info['FPA Pixel Size'] * info['PixelAggregationSize']
        except KeyError:
            # Use pixel units if FPA Pixel Size is not known
            px_size = 1
        x_locs = np.linspace(
            0, X.shape[1]*px_size, num=X.shape[1], endpoint=False)
        y_locs = np.linspace(
            0, X.shape[0]*px_size, num=X.shape[0], endpoint=False)

        features, data, additional_table = _spectra_from_image(
            X, features, x_locs, y_locs)

        import_params = ['Effective Laser Wavenumber',
                         'Under Sampling Ratio',
                         ]
        new_attributes = []
        new_columns = []
        for param_key in import_params:
            try:
                param = info[param_key]
            except KeyError:
                pass
            else:
                new_attributes.append(ContinuousVariable.make(param_key))
                new_columns.append(np.full((len(data),), param))

        domain = Domain(additional_table.domain.attributes,
                        additional_table.domain.class_vars,
                        additional_table.domain.metas + tuple(new_attributes))
        table = additional_table.transform(domain)
        table[:, new_attributes] = np.asarray(new_columns).T

        return (features, data, table)
