"""app.landsat: handle request for Landsat-tiler."""

from typing import Any, Dict, Tuple, Union
from typing.io import BinaryIO

import json
import urllib
import os
import multiprocessing
import numpy
from functools import partial
from concurrent import futures
from skimage import exposure

import mercantile
import rasterio
from rasterio import warp
from rio_tiler import landsat8, utils
from rio_tiler.mercator import get_zooms
from rio_tiler.profiles import img_profiles
from rio_tiler.utils import array_to_image, get_colormap, expression
from rio_toa import reflectance, brightness_temp, toa_utils

from rio_tiler_mvt.mvt import encoder as mvtEncoder

from remotepixel_tiler.utils import _postprocess

from lambda_proxy.proxy import API

APP = API(name="landsat-tiler")
LANDSAT_BUCKET = "s3://landsat-pds"
LANDSAT_BANDS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "QA"]

MAX_THREADS = int(os.environ.get("MAX_THREADS", multiprocessing.cpu_count() * 5))

def landsat8_tile(sceneid, tile_x, tile_y, tile_z, bands=("4", "3", "2"), tilesize=256, pan=False, percents="", **kwargs):
    """
    Create mercator tile from Landsat-8 data.

    Attributes
    ----------
    sceneid : str
        Landsat sceneid. For scenes after May 2017,
        sceneid have to be LANDSAT_PRODUCT_ID.
    tile_x : int
        Mercator tile X index.
    tile_y : int
        Mercator tile Y index.
    tile_z : int
        Mercator tile ZOOM level.
    bands : tuple, str, optional (default: ("4", "3", "2"))
        Bands index for the RGB combination.
    tilesize : int, optional (default: 256)
        Output image size.
    pan : boolean, optional (default: False)
        If True, apply pan-sharpening.
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
        if band not in LANDSAT_BANDS:
            raise InvalidBandName("{} is not a valid Landsat band name".format(band))

    scene_params = landsat8._landsat_parse_scene_id(sceneid)
    meta_data = landsat8._landsat_get_mtl(sceneid).get("L1_METADATA_FILE")
    landsat_address = "{}/{}".format(LANDSAT_BUCKET, scene_params["key"])

    wgs_bounds = toa_utils._get_bounds_from_metadata(meta_data["PRODUCT_METADATA"])

    addresses = [
        "{}_B{}.TIF".format(landsat_address, band) for band in bands
    ]

    values = []
    percents = percents.split(',')
    i = 0
    for address in addresses:
        with rasterio.open(address) as src:
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

    if not utils.tile_exists(wgs_bounds, tile_z, tile_x, tile_y):
        # raise TileOutsideBounds(
        #     "Tile {}/{}/{} is outside image bounds".format(tile_z, tile_x, tile_y)
        # )
        return (
            "OK",
            f"image/{ext}",
            b'',
        )

    mercator_tile = mercantile.Tile(x=tile_x, y=tile_y, z=tile_z)
    tile_bounds = mercantile.xy_bounds(mercator_tile)

    def _tiler(band):
        address = "{}_B{}.TIF".format(landsat_address, band)
        if band == "QA":
            nodata = 1
        else:
            nodata = 0

        return utils.tile_read(
            address, bounds=tile_bounds, tilesize=tilesize, nodata=nodata, **kwargs
        )

    with futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        data, masks = zip(*list(executor.map(_tiler, bands)))
        mask = numpy.all(masks, axis=0).astype(numpy.uint8) * 255
        
    new_data = list(data)
    has_modification = False
    for ds in range(0, len(new_data)):
        if values[ds][0] is not None and values[ds][1] is not None:
            has_modification = True
            new_data[ds] = exposure.rescale_intensity(new_data[ds], in_range=(values[ds][0], values[ds][1]), out_range=(0,255))
    if has_modification == True:
        data = numpy.array(new_data).astype(numpy.uint8)

    data = numpy.concatenate(data)

    if pan:
        pan_address = "{}_B8.TIF".format(landsat_address)
        matrix_pan, mask = utils.tile_read(
            pan_address, tile_bounds, tilesize, nodata=0
        )
        data = utils.pansharpening_brovey(data, matrix_pan, 0.2, matrix_pan.dtype)

    sun_elev = meta_data["IMAGE_ATTRIBUTES"]["SUN_ELEVATION"]

    for bdx, band in enumerate(bands):
        if band in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:  # OLI
            multi_reflect = meta_data["RADIOMETRIC_RESCALING"].get(
                "REFLECTANCE_MULT_BAND_{}".format(band)
            )

            add_reflect = meta_data["RADIOMETRIC_RESCALING"].get(
                "REFLECTANCE_ADD_BAND_{}".format(band)
            )

            data[bdx] = 10000 * reflectance.reflectance(
                data[bdx], multi_reflect, add_reflect, sun_elev
            )

        elif band in ["10", "11"]:  # TIRS
            multi_rad = meta_data["RADIOMETRIC_RESCALING"].get(
                "RADIANCE_MULT_BAND_{}".format(band)
            )

            add_rad = meta_data["RADIOMETRIC_RESCALING"].get(
                "RADIANCE_ADD_BAND_{}".format(band)
            )

            k1 = meta_data["TIRS_THERMAL_CONSTANTS"].get(
                "K1_CONSTANT_BAND_{}".format(band)
            )

            k2 = meta_data["TIRS_THERMAL_CONSTANTS"].get(
                "K2_CONSTANT_BAND_{}".format(band)
            )

            data[bdx] = brightness_temp.brightness_temp(
                data[bdx], multi_rad, add_rad, k1, k2
            )

    return data, mask


class LandsatTilerError(Exception):
    """Base exception class."""


@APP.route(
    "/<sceneid>.json",
    methods=["GET"],
    cors=True,
    payload_compression_method="gzip",
    binary_b64encode=True,
    ttl=3600,
    tag=["metadata"],
)
@APP.route(
    "/tilejson.json",
    methods=["GET"],
    cors=True,
    payload_compression_method="gzip",
    binary_b64encode=True,
    ttl=3600,
    tag=["metadata"],
)
@APP.pass_event
def tilejson_handler(
    event: Dict,
    sceneid: str,
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
    if tile_format in ["pbf", "mvt"]:
        tile_url = f"{APP.host}/tiles/{sceneid}/{{z}}/{{x}}/{{y}}.{tile_format}?{qs}"
    else:
        tile_url = f"{APP.host}/tiles/{sceneid}/{{z}}/{{x}}/{{y}}@{tile_scale}x.{tile_format}?{qs}"

    scene_params = landsat8._landsat_parse_scene_id(sceneid)
    landsat_address = f"{LANDSAT_BUCKET}/{scene_params['key']}_BQA.TIF"
    with rasterio.open(landsat_address) as src_dst:
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
        name=sceneid,
        tilejson="2.1.0",
        tiles=[tile_url],
    )
    return ("OK", "application/json", json.dumps(meta))


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
    return ("OK", "application/json", json.dumps(landsat8.bounds(scene)))


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
    info = landsat8.metadata(scene, pmin, pmax)
    return ("OK", "application/json", json.dumps(info))


@APP.route(
    "/tiles/<scene>/<int:z>/<int:x>/<int:y>.pbf",
    methods=["GET"],
    cors=True,
    token=True,
    payload_compression_method="gzip",
    binary_b64encode=True,
    ttl=3600,
    tag=["tiles"],
)
def mvttiles(
    scene: str,
    z: int,
    x: int,
    y: int,
    bands: str = None,
    tile_size: Union[str, int] = 256,
    pixel_selection: str = "first",
    feature_type: str = "point",
    resampling_method: str = "nearest",
) -> Tuple[str, str, BinaryIO]:
    """Handle MVT tile requests."""
    if tile_size is not None and isinstance(tile_size, str):
        tile_size = int(tile_size)

    bands = tuple(bands.split(","))
    tile, mask = landsat8.tile(scene, x, y, z, bands=bands, tilesize=tile_size)

    band_descriptions = [f"Band_{b}" for b in bands]
    return (
        "OK",
        "application/x-protobuf",
        mvtEncoder(tile, mask, band_descriptions, "landsat", feature_type=feature_type),
    )


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
def tiles(
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
    pan: bool = False,
) -> Tuple[str, str, BinaryIO]:
    """Handle tile requests."""
    driver = "jpeg" if ext == "jpg" else ext

    if bands and expr:
        raise LandsatTilerError("Cannot pass bands and expression")

    tilesize = scale * 256

    pan = True if pan else False
    if expr is not None:
        tile, mask = expression(scene, x, y, z, expr=expr, tilesize=tilesize, pan=pan)

    elif bands is not None:
        tile, mask = landsat8_tile(
            scene, x, y, z, bands=tuple(bands.split(",")), tilesize=tilesize, pan=pan, percents=percents
        )
    else:
        raise LandsatTilerError("No bands nor expression given")

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
