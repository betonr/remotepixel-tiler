"""app.cbers: handle request for CBERS-tiler."""

import os
import re
import json
import multiprocessing
import numpy
from functools import partial
from concurrent import futures

from typing import BinaryIO, Tuple, Union

import mercantile
import rasterio
from rasterio.warp import transform_bounds

from rio_tiler import cbers, utils
from rio_tiler.profiles import img_profiles
from rio_tiler.utils import array_to_image, get_colormap, expression, linear_rescale, _chunks
from rio_tiler.errors import TileOutsideBounds, InvalidBandName, InvalidCBERSSceneId
from aws_sat_api.search import cbers as cbers_search

from rio_color.operations import parse_operations
from rio_color.utils import scale_dtype, to_math_type

from remotepixel_tiler.utils import _postprocess, rescale_intensity

from lambda_proxy.proxy import API

APP = API(name="cbers-tiler")

# CBERS
CBERS_BUCKET = "s3://cbers-pds"
MAX_THREADS = int(os.environ.get("MAX_THREADS", multiprocessing.cpu_count() * 5))


def cbers_tile(sceneid, tile_x, tile_y, tile_z, bands, tilesize=256, percents='', **kwargs):
    """
    Create mercator tile from CBERS data.

    Attributes
    ----------
    sceneid : str
        CBERS sceneid.
    tile_x : int
        Mercator tile X index.
    tile_y : int
        Mercator tile Y index.
    tile_z : int
        Mercator tile ZOOM level.
    bands : tuple, int, optional (default: None)
        Bands index for the RGB combination. If None uses default
        defined for the instrument
    tilesize : int, optional (default: 256)
        Output image size.
    percents : str 
        parcents to apply in bands (linear)
    kwargs: dict, optional
        These will be passed to the 'rio_tiler.utils._tile_read' function.

    Returns
    -------
    data : numpy ndarray
    mask: numpy array

    """
    scene_params = cbers._cbers_parse_scene_id(sceneid)

    if not bands:
        bands = scene_params["rgb"]

    if not isinstance(bands, tuple):
        bands = tuple((bands,))

    for band in bands:
        if band not in scene_params["bands"]:
            raise InvalidBandName(
                "{} is not a valid band name for {} CBERS instrument".format(
                    band, scene_params["instrument"]
                )
            )

    cbers_address = "{}/{}".format(CBERS_BUCKET, scene_params["key"])

    addresses = [
        "{}/{}_BAND{}.tif".format(cbers_address, sceneid, band) for band in bands
    ]

    values = []
    percents = percents.split(',')
    i = 0
    for address in addresses:
        with rasterio.open(address) as src:
            bounds = transform_bounds(src.crs, "epsg:4326", *src.bounds, densify_pts=21)
            if int(percents[i]) != 0 and int(percents[i+1]) != 100:
                overviews = src.overviews(1)
                if len(overviews) > 0:
                    d = src.read( out_shape=(1, int(src.height / overviews[len(overviews)-1]), int(src.width / overviews[len(overviews)-1]) ))
                else:
                    d = src.read()

                dflatten = numpy.array(d.flatten())
                p_start, p_end = numpy.percentile( dflatten[dflatten>0], (int(percents[i]), (int(percents[i+1]))) )
                values.append([p_start, p_end])
            else:
                values.append([None, None])
            i += 2

    if not utils.tile_exists(bounds, tile_z, tile_x, tile_y):
        # raise TileOutsideBounds(
        #     "Tile {}/{}/{} is outside image bounds".format(tile_z, tile_x, tile_y)
        # )
        return None, None

    mercator_tile = mercantile.Tile(x=tile_x, y=tile_y, z=tile_z)
    tile_bounds = mercantile.xy_bounds(mercator_tile)

    _tiler = partial(
        utils.tile_read, bounds=tile_bounds, tilesize=tilesize, nodata=0, **kwargs
    )
    with futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        data, masks = zip(*list(executor.map(_tiler, addresses)))
        mask = numpy.all(masks, axis=0).astype(numpy.uint8) * 255

    new_data = list(data)
    has_modification = False
    for ds in range(0, len(new_data)):
        if values[ds][0] is not None and values[ds][1] is not None:
            has_modification = True
            new_data[ds] = rescale_intensity(new_data[ds], in_range=(values[ds][0], values[ds][1]), out_range=(0,255))
    if has_modification == True:
        data = numpy.array(new_data).astype(numpy.uint8)

    return numpy.concatenate(data), mask


class CbersTilerError(Exception):
    """Base exception class."""


@APP.route(
    "/search/<string:path>/<string:row>",
    methods=["GET"],
    cors=True,
    token=True,
    payload_compression_method="gzip",
    binary_b64encode=True,
    tag=["search"],
)
def search(path: str, row: str) -> Tuple[str, str, str]:
    """Handle search requests."""
    data = list(cbers_search(path, row))
    info = {
        "request": {"path": path, "row": row},
        "meta": {"found": len(data)},
        "results": data,
    }
    return ("OK", "application/json", json.dumps(info))


@APP.route(
    "/bounds/<scene>",
    methods=["GET"],
    cors=True,
    token=True,
    payload_compression_method="gzip",
    binary_b64encode=True,
    ttl=3600,
    tag=["metadata"],
)
def bounds(scene: str) -> Tuple[str, str, str]:
    """Handle bounds requests."""
    return ("OK", "application/json", json.dumps(cbers.bounds(scene)))


@APP.route(
    "/metadata/<scene>",
    methods=["GET"],
    cors=True,
    token=True,
    payload_compression_method="gzip",
    binary_b64encode=True,
    ttl=3600,
    tag=["metadata"],
)
def metadata(
    scene: str, pmin: Union[str, float] = 2., pmax: Union[str, float] = 98.
) -> Tuple[str, str, str]:
    """Handle metadata requests."""
    pmin = float(pmin) if isinstance(pmin, str) else pmin
    pmax = float(pmax) if isinstance(pmax, str) else pmax
    info = cbers.metadata(scene, pmin, pmax)
    return ("OK", "application/json", json.dumps(info))


@APP.route(
    "/tiles/<scene>/<int:z>/<int:x>/<int:y>.<ext>",
    methods=["GET"],
    cors=True,
    token=True,
    payload_compression_method="gzip",
    binary_b64encode=True,
    ttl=3600,
    tag=["tiles"],
)
@APP.route(
    "/tiles/<scene>/<int:z>/<int:x>/<int:y>@<int:scale>x.<ext>",
    methods=["GET"],
    cors=True,
    token=True,
    payload_compression_method="gzip",
    binary_b64encode=True,
    ttl=3600,
    tag=["tiles"],
)
def tile(
    scene: str,
    z: int,
    x: int,
    y: int,
    scale: int = 1,
    ext: str = "png",
    bands: str = None,
    percents: str = "",
    expr: str = None,
    rescale: str = None,
    color_formula: str = None,
    color_map: str = None,
) -> Tuple[str, str, BinaryIO]:
    """Handle tile requests."""
    driver = "jpeg" if ext == "jpg" else ext

    if bands and expr:
        raise CbersTilerError("Cannot pass bands and expression")

    tilesize = scale * 256

    if expr is not None:
        tile, mask = expression(scene, x, y, z, expr=expr, tilesize=tilesize)
    elif bands is not None:
        tile, mask = cbers_tile(
            scene, x, y, z, bands=tuple(bands.split(",")), tilesize=tilesize, percents=percents
        )
    else:
        raise CbersTilerError("No bands nor expression given")

    if tile is None or mask is None:
        return (
            "OK",
            f"image/png",
            b'',
        )

    rtile, rmask = _postprocess(
        tile, mask, rescale=None, color_formula=color_formula
    )

    if color_map:
        color_map = get_colormap(color_map, format="gdal")

    options = img_profiles.get(driver, {})
    return (
        "OK",
        f"image/{ext}",
        array_to_image(rtile, rmask, img_format=driver, color_map=color_map, **options),
    )


@APP.route("/favicon.ico", methods=["GET"], cors=True, tag=["other"])
def favicon() -> Tuple[str, str, str]:
    """Favicon."""
    return ("EMPTY", "text/plain", "")
