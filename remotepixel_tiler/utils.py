"""Utility functions."""

from typing import Tuple

import numpy

from rio_color.operations import parse_operations
from rio_color.utils import scale_dtype, to_math_type

from rio_tiler.utils import linear_rescale, _chunks


def _postprocess(
    tile: numpy.ndarray,
    mask: numpy.ndarray,
    rescale: str = None,
    color_formula: str = None,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    if rescale:
        rescale_arr = list(map(float, rescale.split(",")))
        rescale_arr = list(_chunks(rescale_arr, 2))
        if len(rescale_arr) != tile.shape[0]:
            rescale_arr = ((rescale_arr[0]),) * tile.shape[0]
        for bdx in range(tile.shape[0]):
            tile[bdx] = numpy.where(
                mask,
                linear_rescale(
                    tile[bdx], in_range=rescale_arr[bdx], out_range=[0, 255]
                ),
                0,
            )
        tile = tile.astype(numpy.uint8)

    if color_formula:
        # make sure one last time we don't have
        # negative value before applying color formula
        tile[tile < 0] = 0
        for ops in parse_operations(color_formula):
            tile = scale_dtype(ops(to_math_type(tile)), numpy.uint8)

    return tile, mask


def intensity_range(image, range_values='image', clip_negative=False):
    """Return image intensity range (min, max) based on desired value type.
    Parameters
    ----------
    image : array
        Input image.
    range_values : str or 2-tuple, optional
        The image intensity range is configured by this parameter.
        The possible values for this parameter are enumerated below.
        'image'
            Return image min/max as the range.
        'dtype'
            Return min/max of the image's dtype as the range.
        dtype-name
            Return intensity range based on desired `dtype`. Must be valid key
            in `DTYPE_RANGE`. Note: `image` is ignored for this range type.
        2-tuple
            Return `range_values` as min/max intensities. Note that there's no
            reason to use this function if you just want to specify the
            intensity range explicitly. This option is included for functions
            that use `intensity_range` to support all desired range types.
    clip_negative : bool, optional
        If True, clip the negative range (i.e. return 0 for min intensity)
        even if the image dtype allows negative values.
    """
    _integer_types = (numpy.byte, numpy.ubyte,      # 8 bits
                  numpy.short, numpy.ushort,        # 16 bits
                  numpy.intc, numpy.uintc,          # 16 or 32 or 64 bits
                  numpy.int_, numpy.uint,           # 32 or 64 bits
                  numpy.longlong, numpy.ulonglong)  # 64 bits
    _integer_ranges = {t: (numpy.iinfo(t).min, numpy.iinfo(t).max)
                   for t in _integer_types}
    dtype_range = {numpy.bool_: (False, True),
               numpy.bool8: (False, True),
               numpy.float16: (-1, 1),
               numpy.float32: (-1, 1),
               numpy.float64: (-1, 1)}
    dtype_range.update(_integer_ranges)

    DTYPE_RANGE = dtype_range.copy()
    DTYPE_RANGE.update((d.__name__, limits) for d, limits in dtype_range.items())
    DTYPE_RANGE.update({'uint10': (0, 2 ** 10 - 1),
                        'uint12': (0, 2 ** 12 - 1),
                        'uint14': (0, 2 ** 14 - 1),
                        'bool': dtype_range[numpy.bool_],
                        'float': dtype_range[numpy.float64]})

    if range_values == 'dtype':
        range_values = image.dtype.type

    if range_values == 'image':
        i_min = numpy.min(image)
        i_max = numpy.max(image)
    elif range_values in DTYPE_RANGE:
        i_min, i_max = DTYPE_RANGE[range_values]
        if clip_negative:
            i_min = 0
    else:
        i_min, i_max = range_values
    return i_min, i_max


def rescale_intensity(image, in_range='image', out_range='dtype'):
    """Return image after stretching or shrinking its intensity levels.
    The desired intensity range of the input and output, `in_range` and
    `out_range` respectively, are used to stretch or shrink the intensity range
    of the input image. See examples below.
    Parameters
    ----------
    image : array
        Image array.
    in_range, out_range : str or 2-tuple, optional
        Min and max intensity values of input and output image.
        The possible values for this parameter are enumerated below.
        'image'
            Use image min/max as the intensity range.
        'dtype'
            Use min/max of the image's dtype as the intensity range.
        dtype-name
            Use intensity range based on desired `dtype`. Must be valid key
            in `DTYPE_RANGE`.
        2-tuple
            Use `range_values` as explicit min/max intensities.
    Returns
    -------
    out : array
        Image array after rescaling its intensity. This image is the same dtype
        as the input image.
    See Also
    --------
    equalize_hist
    Examples
    --------
    By default, the min/max intensities of the input image are stretched to
    the limits allowed by the image's dtype, since `in_range` defaults to
    'image' and `out_range` defaults to 'dtype':
    >>> image = np.array([51, 102, 153], dtype=np.uint8)
    >>> rescale_intensity(image)
    array([  0, 127, 255], dtype=uint8)
    It's easy to accidentally convert an image dtype from uint8 to float:
    >>> 1.0 * image
    array([ 51., 102., 153.])
    Use `rescale_intensity` to rescale to the proper range for float dtypes:
    >>> image_float = 1.0 * image
    >>> rescale_intensity(image_float)
    array([0. , 0.5, 1. ])
    To maintain the low contrast of the original, use the `in_range` parameter:
    >>> rescale_intensity(image_float, in_range=(0, 255))
    array([0.2, 0.4, 0.6])
    If the min/max value of `in_range` is more/less than the min/max image
    intensity, then the intensity levels are clipped:
    >>> rescale_intensity(image_float, in_range=(0, 102))
    array([0.5, 1. , 1. ])
    If you have an image with signed integers but want to rescale the image to
    just the positive range, use the `out_range` parameter:
    >>> image = np.array([-10, 0, 10], dtype=np.int8)
    >>> rescale_intensity(image, out_range=(0, 127))
    array([  0,  63, 127], dtype=int8)
    """
    dtype = image.dtype.type

    imin, imax = intensity_range(image, in_range)
    omin, omax = intensity_range(image, out_range, clip_negative=(imin >= 0))

    # Fast test for multiple values, operations with at least 1 NaN return NaN
    if numpy.isnan(imin + imax + omin + omax):
        return
    image = numpy.clip(image, imin, imax)

    if imin != imax:
        image = (image - imin) / float(imax - imin)
    return numpy.asarray(image * (omax - omin) + omin, dtype=dtype)