"""app.sentinel: handle request for Sentinel-tiler."""

from typing import Any, Dict, Tuple, Union, BinaryIO

import os
import re
import json
import multiprocessing
import numpy
from functools import partial
from concurrent import futures
from skimage import exposure
import urllib

import mercantile
import rasterio
from rasterio import warp
from rio_tiler import sentinel2, utils
from rio_tiler.mercator import get_zooms
from rio_tiler.profiles import img_profiles
from rio_tiler.utils import array_to_image, get_colormap, expression
from rio_tiler.errors import TileOutsideBounds, InvalidBandName, InvalidSentinelSceneId

from remotepixel_tiler.utils import _postprocess

from lambda_proxy.proxy import API

APP = API(name="sentinel-tiler")
SENTINEL_BUCKET = "s3://sentinel-s2-l1c"
SENTINEL_BANDS = ["01", "02", "03", "04", "05", "06", "07", "08", "8A", "09", "10", "11", "12"]
MAX_THREADS = int(os.environ.get("MAX_THREADS", multiprocessing.cpu_count() * 5))

def sentinel2_tile(sceneid, tile_x, tile_y, tile_z, bands=("04", "03", "02"), tilesize=256, percents='', **kwargs):
    """
    Create mercator tile from Sentinel-2 data.

    Attributes
    ----------
    sceneid : str
        Sentinel-2 sceneid.
    tile_x : int
        Mercator tile X index.
    tile_y : int
        Mercator tile Y index.
    tile_z : int
        Mercator tile ZOOM level.
    bands : tuple, str, optional (default: ('04', '03', '02'))
        Bands index for the RGB combination.
    tilesize : int, optional (default: 256)
        Output image size.
    kwargs: dict, optional
        These will be passed to the 'rio_tiler.utils._tile_read' function.

    Returns
    -------
    data : numpy ndarray
    mask: numpy array

    """
    if not isinstance(bands, tuple):
        bands = tuple((bands,))

    for band in bands:
        if band not in SENTINEL_BANDS:
            raise InvalidBandName("{} is not a valid Sentinel band name".format(band))

    scene_params = sentinel2._sentinel_parse_scene_id(sceneid)
    sentinel_address = "{}/{}".format(SENTINEL_BUCKET, scene_params["key"])

    addresses = ["{}/B{}.jp2".format(sentinel_address, band) for band in bands]

    values = []
    percents = percents.split(',')
    i = 0
    for address in addresses:
        with rasterio.open(address) as src:
            bounds = warp.transform_bounds(src.crs, "epsg:4326", *src.bounds, densify_pts=21)
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
        return (
            "OK",
            f"image/png",
            b'',
        )

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
            new_data[ds] = exposure.rescale_intensity(new_data[ds], in_range=(values[ds][0], values[ds][1]), out_range=(0,255))
    if has_modification == True:
        data = numpy.array(new_data).astype(numpy.uint8)

    return numpy.concatenate(data), mask


class SentinelTilerError(Exception):
    """Base exception class."""


@APP.route(
    "/s2/<scene>.json",
    methods=["GET"],
    cors=True,
    token=True,
    payload_compression_method="gzip",
    binary_b64encode=True,
    ttl=3600,
    tag=["metadata"],
)
@APP.route(
    "/s2/tilejson.json",
    methods=["GET"],
    cors=True,
    token=True,
    payload_compression_method="gzip",
    binary_b64encode=True,
    ttl=3600,
    tag=["metadata"],
)
@APP.pass_event
def tilejson_handler(
    event: Dict,
    scene: str,
    tile_format: str = "png",
    tile_scale: int = 1,
    **kwargs: Any,
) -> Tuple[str, str, str]:
    """Handle /tilejson.json requests."""
    # HACK
    token = event["multiValueQueryStringParameters"].get("access_token")
    if token:
        kwargs.update(dict(access_token=token[0]))

    qs = urllib.parse.urlencode(list(kwargs.items()))
    tile_url = f"{APP.host}/s2/tiles/{scene}/{{z}}/{{x}}/{{y}}@{tile_scale}x.{tile_format}?{qs}"

    scene_params = sentinel2._sentinel_parse_scene_id(scene)
    sentinel_address = "s3://{}/{}/B{}.jp2".format(
        sentinel2.SENTINEL_BUCKET, scene_params["key"], "04"
    )
    with rasterio.open(sentinel_address) as src_dst:
        bounds = warp.transform_bounds(
            *[src_dst.crs, "epsg:4326"] + list(src_dst.bounds), densify_pts=21
        )
        minzoom, maxzoom = get_zooms(src_dst)
        center = [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2, minzoom]

    meta = dict(
        bounds=bounds,
        center=center,
        minzoom=minzoom,
        maxzoom=maxzoom,
        name=scene,
        tilejson="2.1.0",
        tiles=[tile_url],
    )
    return ("OK", "application/json", json.dumps(meta))


@APP.route(
    "/s2/bounds/<scene>",
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
    return ("OK", "application/json", json.dumps(sentinel2.bounds(scene)))


@APP.route(
    "/s2/metadata/<scene>",
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
    info = sentinel2.metadata(scene, pmin, pmax)
    return ("OK", "application/json", json.dumps(info))


@APP.route(
    "/s2/tiles/<scene>/<int:z>/<int:x>/<int:y>.<ext>",
    methods=["GET"],
    cors=True,
    token=True,
    payload_compression_method="gzip",
    binary_b64encode=True,
    ttl=3600,
    tag=["tiles"],
)
@APP.route(
    "/s2/tiles/<scene>/<int:z>/<int:x>/<int:y>@<int:scale>x.<ext>",
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
        raise SentinelTilerError("Cannot pass bands and expression")

    tilesize = scale * 256

    if expr is not None:
        tile, mask = expression(scene, x, y, z, expr, tilesize=tilesize)

    elif bands is not None:
        tile, mask = sentinel2_tile(
            scene, x, y, z, bands=tuple(bands.split(",")), tilesize=tilesize, percents=percents
        )
    else:
        raise SentinelTilerError("No bands nor expression given")

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
