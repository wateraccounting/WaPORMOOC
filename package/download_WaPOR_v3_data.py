"""
This code is used to download and preprocess WaPOR v3 data.
It can retrieve point data for one or multiple locations, or
download raster datasets and generate netCDF files per variable for a defined area of interest.

The code is still under development, and the functions still need proper documentation.

Code by: Solomon Seyoum (sd.seyoum7@gmail.com)
for any bug please contact @ sd.seyoum7@gmail.com.
"""

# import osgeo
import glob
import multiprocessing as mp
import os
import re
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd
import psutil
import rasterio
import requests
import rioxarray as rio
import xarray as xr
from dask.distributed import Client, LocalCluster, get_client
from dateutil import parser
from osgeo import gdal
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import from_bounds
from shapely.geometry import box

# Constants
GDAL_ENV_VARS = {
    # "GDAL_DISABLE_READDIR_ON_OPEN": "TRUE",
    "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
    "GDAL_SWATH_SIZE": "200000000",
    "VSI_CURL_CACHE_SIZE": "200000000",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",  #  critical for cloud speed
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "TIF",
    "GDAL_HTTP_MULTIRANGE": "YES",  # parallel HTTP range requests
    "GDAL_MEM_ENABLE_OPEN": "YES",
}
FREQUENCY_SYMBOLS = {"daily": "E", "dekadal": "D", "monthly": "M", "yearly": "A"}
# units = {
#     "GBWP": "kg/m³",
#     "NBWP": "kg/m³",
#     "TBP": "kg/ha",
#     "NPP": "gC/m²/day",
#     "AETI": "mm/day",
#     "PCP": "mm/day",
#     "ET": "mm/day",
#     "T": "mm/day",
#     "I": "mm/day",
#     "RSM": "-",
#     "LCC": "-",
# }


def initialize_multiprocessing():
    """
    Initializes and returns a Dask client for parallel processing.

    This function follows a robust pattern:
    1. Tries to get a handle to an already existing Dask client.
    2. If none exists, it tries to connect to a Dask scheduler running on the
       local machine at the default port.
    3. If no existing scheduler is found, it starts a new LocalCluster using
       a portion of the available CPU cores.

    Returns:
        dask.distributed.Client: An active Dask client.
    """
    try:
        # Try to get an existing client without raising an error if none exists.
        client = get_client(timeout="2s")
        print("Reusing existing Dask client.")
        return client
    except ValueError:
        # No existing client found, proceed to the next step.
        pass

    try:
        # Try connecting to an existing scheduler at the default address.
        client = Client("tcp://localhost:8786", timeout="4s")
        print("Connected to existing Dask scheduler at localhost:8786.")
        return client
    except (OSError, TimeoutError):
        # No running Dask scheduler found, start a new local one.
        print("No running Dask scheduler found, starting a new LocalCluster...")

        # Use a standard temporary directory for Dask worker files.
        tmp_dir = os.environ.get("TEMP", "/tmp")

        total_memory = psutil.virtual_memory().total  # bytes
        # Use 85% of total memory as limit per worker
        memory_limit_bytes = int(0.85 * total_memory)
        memory_limit = f"{memory_limit_bytes // (1024**3)}GB"  # Convert to GB

        cluster = LocalCluster(
            # Let Dask choose the best IP address.
            ip="",
            # Use 90% of available CPU cores to leave resources for other tasks.
            n_workers=max(1, int(0.9 * mp.cpu_count())),
            # processes=False uses threads, which is efficient for NumPy/Xarray
            # operations that release the Python Global Interpreter Lock (GIL).
            processes=False,
            local_directory=tmp_dir,
            # WARNING: Hardcoded memory limit. This might fail on machines with less RAM.
            # It's often better to let Dask manage memory automatically.
            memory_limit=memory_limit,
        )
        client = Client(cluster)
        print(
            f"Started new LocalCluster with {len(client.scheduler_info()['workers'])} workers."
        )
        return client


def set_gdal_environment():
    os.environ.update(GDAL_ENV_VARS)


def Open_with_rasterio(href, xmin, ymin, xmax, ymax):
    with rasterio.open(href) as src:
        # Compute window of only the bbox
        window = from_bounds(xmin, ymin, xmax, ymax, src.transform)

        # Read only that chunk
        data = src.read(1, window=window)
        # Update transform for clipped raster
        # transform = src.window_transform(window)
        a = src.window_transform(window)
        xs = a.c + (np.arange(data.shape[1]) + 0.5) * a.a
        ys = a.f + (np.arange(data.shape[0]) + 0.5) * a.e

        da = (
            xr.DataArray(data, dims=("y", "x"), coords={"y": ys, "x": xs})
            .rio.set_spatial_dims("x", "y")
            .rio.write_crs(src.crs)
            .rio.write_nodata(src.nodata)
        )
    return da


# get dekadal timesteps
def get_dekadal_timestep(begin, end):
    dtrange = pd.date_range(begin, end)
    d = dtrange.day - np.clip((dtrange.day - 1) // 10, 0, 2) * 10 - 1
    date = dtrange.values - np.array(d, dtype="timedelta64[D]")
    return np.unique(date)


def get_location_L3_v3(L3_code, location_code):
    mosaic_url = f"{base_url_v3()}/mosaicsets/{L3_code}/rasters"

    output = collect_responses(mosaic_url, info=["code", "grid"])

    for i in range(len(output)):
        if output[i][1]["tile"]["code"] == location_code:
            return output[i][1]["tile"]["caption"]
    return None


def get_country_code_for_L3_v3(l3url, aoi_shp):
    from shapely.geometry import box

    aoi = box(*aoi_shp.geometry.total_bounds)  # area of interest bounding box polygon
    shp_crs_epsg = aoi_shp.crs.to_epsg()

    for L3_url in l3url:
        ds = rio.open_rasterio(L3_url)
        L3_bbox = box(*ds.rio.transform_bounds(f"EPSG:{shp_crs_epsg}"))

        if L3_bbox.intersects(aoi):
            L3_code = os.path.basename(L3_url).split(".")[1]
            location_code = os.path.basename(L3_url).split(".")[2]
            location = get_location_L3_v3(L3_code, location_code)

            if location is not None:
                # print("the AOI is within {0}".format(location))
                return location, location_code, L3_url

    return None  #'AOI is not in the L3 available areas


def get_clipped_ds(da, shape, crs, template, scale_factor):
    clipped = da
    if crs.to_epsg() != 4326:
        clipped = clipped.rio.reproject("EPSG:4326")
        clipped.rio.write_crs("EPSG:4326", inplace=True)

    if (clipped.x != template.x).all() or (clipped.y != template.y).all():
        clipped = clipped.interp({"x": template.x, "y": template.y}, method="nearest")

    clipped = clipped.rio.clip(shape.geometry.values, shape.crs)
    clipped = clipped.rename(x="longitude", y="latitude")

    clipped = clipped.where(clipped != clipped.attrs["_FillValue"])
    if clipped.latitude[-1] < clipped.latitude[0]:
        clipped = clipped.reindex(latitude=clipped.latitude[::-1])

    return clipped * scale_factor


def base_url_v3():
    return "https://data.apps.fao.org/gismgr/api/v2/catalog/workspaces/WAPOR-3"


def collect_responses(url: str, info=["code"]) -> list:
    """Calls GISMGR2.0 API and collects responses.

    Parameters
    ----------
    url : str
        URL to get.
    info : list, optional
        Used to filter the response, set to `None` to keep everything, by default ["code"].

    Returns
    -------
    list
        The responses.
    """
    data = {"links": [{"rel": "next", "href": url}]}
    output = list()
    while "next" in [x["rel"] for x in data["links"]]:
        url_ = [x["href"] for x in data["links"] if x["rel"] == "next"][0]
        try:
            response = requests.get(url_)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            print(f"URL: {url_}")
            # print(" Check the if the variable and frequency is correct for WaPO data")
            break
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}.")
            break

        data = response.json()["response"]
        if isinstance(info, list) and "items" in data.keys():
            output += [tuple(x.get(y) for y in info) for x in data["items"]]
        elif "items" in data.keys():
            output += data["items"]
        else:
            output.append(data)
    if isinstance(info, list):
        try:
            output = sorted(output)
        except TypeError:
            output = output
    return output


def get_unit_and_scale(url, info=["code"]):
    data = {"links": [{"rel": "next", "href": url}]}
    output = list()
    while "next" in [x["rel"] for x in data["links"]]:
        url_ = [x["href"] for x in data["links"] if x["rel"] == "next"][0]
        try:
            response = requests.get(url_)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            break
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}.")
            break
        data = response.json()["response"]
        if isinstance(info, list):
            output += [tuple(data[x] for x in info)]
        else:
            output += []
    return output


def generate_urls_v3(variable: str, l3_region=None, period=None) -> tuple:
    """Find resource URLs for an WaPOR variable for a specified period.

    Parameters
    ----------
    variable : str
        Name of the variable.
    l3_region : _type_, optional
        Three letter code specifying the level-3 region, by default None.
    period : list, optional
        Start and end date in between which resource URLs will be searched, by default None.

    Returns
    -------
    tuple
        Resource URLs.

    Raises
    ------
    ValueError
        Invalid level selected.
    """

    level, _, tres = variable.split("-")

    if (level == "L1") or (level == "L2"):
        base_url = (
            "https://data.apps.fao.org/gismgr/api/v2/catalog/workspaces/WAPOR-3/mapsets"
        )
    elif level == "L3":
        base_url = "https://data.apps.fao.org/gismgr/api/v2/catalog/workspaces/WAPOR-3/mosaicsets"
    else:
        raise ValueError(f"Invalid level {level}.")  # NOTE: TESTED
    mapset_url = f"{base_url}/{variable}"
    # unit = [x[0] for x in collect_responses(mapset_url, info = ["measureUnit"])]
    unit_and_scale = get_unit_and_scale(mapset_url, info=["measureUnit", "scale"])
    mapset_url = f"{base_url}/{variable}/rasters?filter="
    if not isinstance(l3_region, type(None)):
        mapset_url += f"code:CONTAINS:{l3_region};"
    if not isinstance(period, type(None)):
        mapset_url += f"time:OVERLAPS:{period[0]}:{period[1]};"
    urls = [x[0] for x in collect_responses(mapset_url, info=["downloadUrl"])]

    return unit_and_scale, tuple(sorted(urls))


def get_agera5_ET0_PCP(prod, frq, time_info):
    years, months, dekads, dates = time_info

    if frq == "daily":
        dates = dates.astype("datetime64[D]").astype(str)
        if "RET" in prod:
            urls = [
                f"/vsicurl/https://data.apps.fao.org/static/data/c3s/AGERA5_ET0_D/AGERA5_ET0_{dt}.tif"
                for dt in dates
            ]
        elif "PCP" in prod:
            urls = [
                f"/vsicurl/https://data.apps.fao.org/static/data/c3s/MAPSET/AGERA5-PF/C3S.AGERA5-PF.{dt}.tif"
                for dt in dates
            ]
        else:
            print(f"{prod} is not either 'RET or PCP")

    elif frq == "dekadal":
        dekads = dekads.astype("datetime64[D]").astype(str)
        dekads = pd.to_datetime(dekads)
        dks = [
            f"{d.year}-{d.month:02d}-D{1 if d.day <= 10 else 2 if d.day <= 20 else 3}"
            for d in dekads
        ]
        if "RET" in prod:
            urls = [
                f"/vsicurl/https://data.apps.fao.org/static/data/c3s/MAPSET/AGERA5-ET0-D/C3S.AGERA5-ET0-D.{dk}.tif"
                for dk in dks
            ]
        elif "PCP" in prod:
            urls = [
                f"/vsicurl/https://data.apps.fao.org/static/data/c3s/MAPSET/AGERA5-PF-D/C3S.AGERA5-PF-D.{dk}.tif"
                for dk in dks
            ]
        else:
            print(f"{prod} is not either 'RET or PCP")
    elif frq == "monthly":
        months = months.astype("datetime64[D]").astype(str)
        months = pd.to_datetime(months)
        mns = [f"{dt.strftime('%Y-%m')}" for dt in pd.to_datetime(months)]
        if "RET" in prod:
            urls = [
                f"/vsicurl/https://data.apps.fao.org/static/data/c3s/MAPSET/AGERA5-ET0-M/C3S.AGERA5-ET0-M.{mn}.tif"
                for mn in mns
            ]
        elif "PCP" in prod:
            urls = [
                f"/vsicurl/https://data.apps.fao.org/static/data/c3s/MAPSET/AGERA5-PF-M/C3S.AGERA5-PF-M.{mn}.tif"
                for mn in mns
            ]
        else:
            print(f"{prod} is not either 'RET or PCP")

    else:
        years = years.astype("datetime64[D]").astype(str)
        years = pd.to_datetime(years)
        yrs = [x.year for x in years]
        if "RET" in prod:
            urls = [
                f"/vsicurl/https://data.apps.fao.org/static/data/c3s/MAPSET/AGERA5-ET0-A/C3S.AGERA5-ET0-A.{yr}.tif"
                for yr in yrs
            ]
        elif "PCP" in prod:
            urls = [
                f"/vsicurl/https://data.apps.fao.org/static/data/c3s/MAPSET/AGERA5-PF-A/C3S.AGERA5-PF-A.{yr}.tif"
                for yr in yrs
            ]
        else:
            print(f"{prod} is not either 'RET or PCP")

    return urls


def get_shape_and_bbox(data_type, region, crs="EPSG:4326"):
    if isinstance(region, str):
        if not os.path.exists(region):
            raise FileNotFoundError(f"Shapefile not found: {region}")
        gdf = gpd.read_file(region)
        # Ensure CRS is defined
        if gdf.crs is None:
            gdf = gdf.set_crs(crs)
            # drop epty or none geometry
        gdf = gdf[~(gdf.geometry.is_empty | gdf.geometry.isna())]
        # Reproject shapefile to 'EPSG:4326' to match NetCDF data's CRS if different.
        target_crs = "EPSG:4326"
        if gdf.crs is None or gdf.crs != target_crs:
            print(f"Reprojecting shapefile from {gdf.crs} to {target_crs}...")
            gdf = gdf.to_crs(target_crs)
            print("Shapefile reprojected successfully.")

        bbox = box(*gdf.geometry.total_bounds)
        if data_type == "raster":
            buffered_bbox = bbox.buffer(0.02)
        else:
            buffered_bbox = bbox
        bbox_gdf = gpd.GeoDataFrame(geometry=[buffered_bbox], crs="EPSG:4326")

    # Case 2: region is a bounding box (list or tuple of 4 numbers)
    elif isinstance(region, (list, tuple)):
        if len(region) != 4:
            raise ValueError("Bounding box must be (xmin, ymin, xmax, ymax)")
        xmin, ymin, xmax, ymax = map(float, region)
        geom = box(xmin, ymin, xmax, ymax)
        if data_type == "raster":
            buffered_bbox = geom.buffer(0.02)
        else:
            buffered_bbox = geom

        bbox_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs=crs)
        gdf = bbox_gdf
        print(f"Bounding box: {region}")

    else:
        raise TypeError(
            "`region` must be a shapefile path (str) or a bounding box (list/tuple of 4 numbers)"
        )

    return gdf, bbox_gdf.geometry.total_bounds


def get_level3_code(shape):
    vl3 = "L3-AETI-A"
    yr = ["2018-01-01", "2018-12-31"]
    l3url = generate_urls_v3(vl3, l3_region=None, period=yr)
    return get_country_code_for_L3_v3(l3url[1], shape)


def get_template_rst(dir, variables, shape, bbox):
    # Downlaod AETI  raster and clip it so that it can be used as a templat
    temp_file = os.path.join(dir, "template_rst.tif")
    template_var = None
    if not (os.path.exists(temp_file)):
        l3 = [x for x in variables if "L3" in x]
        l2 = [x for x in variables if "L2" in x]
        l1 = [x for x in variables if "L1" in x]
        if len(l3) > 0:
            template_var = l3[0]
        elif len(l2) > 0:
            template_var = l2[0]
        elif len(l1) > 0:
            template_var = l1[0]
        else:
            print("No variabke to be used as a template.")

        if template_var is not None:
            level, var_code, tres = template_var.split("-")
            template = f"{level}-AETI-A"
            print(f"using {template} as template.")
            yr = ["2018-01-01", "2018-12-31"]
            if "L3-" in template:
                location, l3code, tempurl = get_level3_code(shape)
                # unit_ans_scale, urls = generate_urls_v3(template, l3code, yr)
            else:
                unit_ans_scale, urls = generate_urls_v3(
                    template, l3_region=None, period=yr
                )
                tempurl = list(urls)[0]

            if "L3" in template:
                ds = gdal.Open(f"/vsicurl/{tempurl}")
                if ds is None:
                    raise RuntimeError("Could not open raster")

                raster_srs = ds.GetSpatialRef()
                if raster_srs is None:
                    raise ValueError("Raster has no CRS")

                raster_crs = raster_srs.ExportToWkt()
                # print("raster CRS is ", raster_crs)

                # reproject the shape to raster CRS
                shp_proj = shape.to_crs(raster_crs)
                # get new bbox
                bbox_new = shp_proj.geometry.total_bounds

                bands = [1]
                xmin, ymin, xmax, ymax = bbox_new
                bbox_gdal = [xmin, ymax, xmax, ymin]
                translate_options = gdal.TranslateOptions(
                    projWin=bbox_gdal,
                    bandList=bands,
                    creationOptions=["TILED=YES", "COMPRESS=LZW"],
                )
                ds = gdal.Translate(
                    temp_file, f"/vsicurl/{tempurl}", options=translate_options
                )
                ds = ds.FlushCache()
            else:
                bands = [1]
                xmin, ymin, xmax, ymax = bbox
                bbox_gdal = [xmin, ymax, xmax, ymin]
                translate_options = gdal.TranslateOptions(
                    projWin=bbox_gdal,
                    bandList=bands,
                    creationOptions=["TILED=YES", "COMPRESS=LZW"],
                )
                ds = gdal.Translate(
                    temp_file, f"/vsicurl/{tempurl}", options=translate_options
                )
                print(bbox, ds)
                ds = ds.FlushCache()

        # get template from the saved raster
    return get_template(temp_file, shape)


def get_yrs_mns_dekads(period):

    for date_str in period:
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError as e:
            print("Invalid date:", e)
            print(f"{date_str} is not valid. Correct the date and run again.")
            sys.exit()
    start, end = period
    # 1. Generate ranges directly as Pandas DatetimeIndex
    # Note: 'YS' (Year Start) and 'MS' (Month Start) are usually safer for alignment
    years_idx = pd.date_range(start, end, freq="YS")
    months_idx = pd.date_range(start, end, freq="MS")
    dates_idx = pd.date_range(start, end, freq="D")

    # 2. Get dekads (assuming your custom function returns a list/array of dates)
    dekads_raw = get_dekadal_timestep(start, end)

    # 3. Vectorized conversion to numpy datetime64
    # .values returns the underlying numpy array immediately
    years = years_idx.values.astype("datetime64[Y]")
    months = months_idx.values.astype("datetime64[M]")
    dates = dates_idx.values.astype("datetime64[D]")

    # For dekads, we convert the collection once rather than per-item
    dekads = pd.to_datetime(dekads_raw).values.astype("datetime64[D]")

    return years, months, dekads, dates


def get_attributes(prod, frq, name, unit, years, months, dekads):
    # prepare attributes

    # unit = units.get(name, "unknown")

    if "LCC" in prod:
        no_times = years
        attrs = {
            "title": "Land cover class from WaPOR",
            "unit": unit,
            "period": "yearly",
            "quantity": name,
            "source": "WaPOR",
        }
    elif frq == "monthly":
        no_times = months
        attrs = {
            "title": "{0} from WaPOR".format(name),
            "unit": unit,
            "period": "monthly",
            "quantity": name,
            "source": "WaPOR",
        }
    elif frq == "dekadal":
        no_times = dekads
        attrs = {
            "title": "{0} from WaPOR".format(name),
            "unit": unit,
            "period": "dekadal",
            "quantity": name,
            "source": "WaPOR",
        }
    else:
        no_times = years
        attrs = {
            "title": "{0} from WaPOR".format(name),
            "unit": unit,
            "period": "yearly",
            "quantity": name,
            "source": "WaPOR",
        }

    return no_times, attrs


def correct_bbox(variable, url, bbox):
    aoi = box(*bbox)

    if "PCP" in variable or "RET" in variable:
        aoi = aoi.buffer(0.3)

    bbox2 = gpd.GeoDataFrame(crs="epsg:4326", geometry=[aoi])

    with rio.open_rasterio(url) as temp:
        crs = temp.rio.crs

    if crs.to_epsg() != 4326:
        # project the bbox for clipping the dataset to the crs of the da
        f"EPSG:{crs.to_epsg()}"
        bbox_for_clipping = bbox2.to_crs(f"EPSG:{crs.to_epsg()}")
    else:
        bbox_for_clipping = bbox2

    return bbox_for_clipping.geometry.total_bounds, crs


def get_date(s_date):
    try:
        return parser.parse(s_date).date()
    except (ValueError, TypeError):
        raise ValueError(f"Unrecognized date format: {s_date}")


def search_date(filename):
    patterns = [
        r"\d{4}.\d{2}.\d{2}",
        r"\d{4}.\d{2}",
        r"\d{4}$",
        r"\d{4}",
        r"\d{2}$",
        r"\d{4}-\d{2}-\d{2}",
        r"\d{4}-\d{2}",
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group()  # return matched string
    return None  # if no match found


def dekadal2datetime_v3(d):
    yr = int(str(d)[0:4])
    mn = int(str(d)[5:7])
    if str(d)[-2:] == "D1":
        day = "0" + str(1)
    elif str(d)[-2:] == "D2":
        day = 11
    else:
        day = 21

    dstr = "{0}-{1}-{2}".format(yr, mn, day)
    return dstr


def time_index_from_filenames(step, fh):
    """helper function to create a pandas DatetimeIndex
    Filename example: xxx_2015.05.20.tif"""
    filenames = [os.path.splitext(f)[0] for f in fh]
    # print([search_date(f[-10:]) for f in filenames])
    if (step == "dekadal") and ("WAPOR-3" in filenames[0]):
        d = [get_date(dekadal2datetime_v3(f[-10:])) for f in filenames]
        dd = [np.datetime64(i) for i in d]
        # dd = [i.to_numpy().astype("datetime64[D]") for i in d]
        return dd

    else:
        # d = [get_date((search_date(f[-10:]))) for f in filenames]
        d = [pd.to_datetime(search_date(f[-10:])) for f in filenames]

        # dd = [np.datetime64(i) for i in d]
        dd = [i.to_numpy().astype("datetime64[D]") for i in d]
        return dd


def time_index_for_AgERA(time_info, frq):
    years, months, dekads, dates = time_info
    if frq == "daily":
        return dates
    elif frq == "dekadal":
        return dekads
    elif frq == "monthly":
        return months
    else:
        return years


def get_prod_names(prod, FREQUENCY_SYMBOLS):
    if "_" in prod:
        split_str = "_"
    else:
        split_str = "-"

    prod_split = prod.split(".")[0]
    prod_split = prod_split.split(split_str)

    name = prod_split[-2]
    frq = list(FREQUENCY_SYMBOLS.keys())[
        list(FREQUENCY_SYMBOLS.values()).index(prod_split[-1])
    ]
    if "AgERA5" in prod:
        level = None
    else:
        level = int(prod_split[0][1:])
    # print("---prod---\n",prod,  name, frq, level)
    return name, frq, level


def get_template(temp_file, shape):
    with rio.open_rasterio(temp_file) as temp:
        crs = temp.rio.crs
        temp = temp.squeeze(dim="band", drop=True)
        temp.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        target_crs = shape.crs
        temp = temp.rio.write_crs(
            crs, inplace=True
        )  ## Instead of this read the crs from the file
        temp = temp.rio.reproject(target_crs)
        temp = temp.rio.write_crs(target_crs, inplace=True)
        temp = temp.rio.clip(shape.geometry.values, shape.crs, drop=True)
        temp = temp.drop_vars(["spatial_ref"])
        temp = temp.where(temp != temp.attrs["_FillValue"])
        if temp.y[-1] < temp.y[0]:
            temp = temp.reindex(y=temp.y[::-1])
        template = temp
        attrs = temp.attrs
        attrs.update({"crs": "EPSG:4326"})
        template.attrs = attrs
        temp.close()
    template.rio.to_raster(temp_file, compress="ZSTD", zstd_level=22)
    return template


def init_nc(name_nc, dim, var, fill=-9999.0, attr=None):
    # Use context manager to ensure the NetCDF file is properly closed
    with netCDF4.Dataset(name_nc, "w", format="NETCDF4") as out_nc:
        # Add dimensions to nc-file.
        for name, values in dim.items():
            if values is not None:
                # out_nc.createDimension(name, values.size)
                # vals = out_nc.createVariable(name, "f4", (name,), fill_value=fill)
                # vals[:] = values

                out_nc.createDimension(name, values.size)
                vals = out_nc.createVariable(name, "f8", (name,))
                vals[:] = values.astype("float64")
            else:
                out_nc.createDimension(name, None)
                vals = out_nc.createVariable(name, "f4", (name,), fill_value=fill)
                vals.calendar = "standard"
                vals.units = "days since 1970-01-01 00:00"

        # Create CRS variable according to CF conventions (EPSG:4326)
        crs = out_nc.createVariable("crs", "i4")
        crs.long_name = "CRS definition"
        crs.grid_mapping_name = "latitude_longitude"
        crs.epsg_code = "EPSG:4326"
        crs.semi_major_axis = 6378137.0
        crs.inverse_flattening = 298.257223563
        crs.proj4_params = "+proj=longlat +datum=WGS84 +no_defs"

        # Create data variables
        for name, props in var.items():
            vals = out_nc.createVariable(
                props[1]["quantity"],
                "f4",
                props[0],
                zlib=True,
                fill_value=fill,
                complevel=5,  # 9,
                shuffle=True,
                least_significant_digit=1,
                # chunksizes = (1, dim["latitude"].size, dim['longitude'].size),#{"time": 1, "latitude": 500, "longitude": 500}
            )
            vals.setncatts(props[1])
            vals.grid_mapping = "crs"  # <-- The CF-compliant link to the CRS variable

        # Set global attributes if provided
        if attr is not None:
            out_nc.setncatts(attr)


def fill_nc_one_timestep(nc_file, var, time_val=None):

    with netCDF4.Dataset(nc_file, "r+") as out_nc:
        variables = out_nc.variables
        dims = out_nc.dimensions

        # ---- TIME DEPENDENT VARIABLES ----
        if time_val is not None:
            time = variables["time"]
            tidx = time.shape[0]
            time[tidx] = time_val

            for name, field in variables.items():
                if "time" in field.dimensions and name not in dims:
                    # Expected spatial shape (excluding time)
                    expected_shape = tuple(
                        len(dims[d]) for d in field.dimensions if d != "time"
                    )

                    if name in var:
                        data = var[name]

                        # Validate shape
                        if data.shape != expected_shape:
                            raise ValueError(
                                f"Shape mismatch for {name}. "
                                f"Expected {expected_shape}, got {data.shape}"
                            )

                        field[tidx, ...] = data

                    else:
                        # Fill missing variables with fill value
                        dummy = np.full(expected_shape, field._FillValue)
                        field[tidx, ...] = dummy

        # ---- INVARIANT VARIABLES ----
        else:
            for name, data in var.items():
                field = variables[name]

                expected_shape = tuple(len(dims[d]) for d in field.dimensions)

                if data.shape != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for {name}. "
                        f"Expected {expected_shape}, got {data.shape}"
                    )

                field[...] = data


def check_exsting_netcdf_file(nc_dir):
    """
    Checks for an existing NetCDF file, determines the last time step for ach variable.
    """
    last_timestep = None
    var_ts = {}
    time_step = None
    nc_files = glob.glob(os.path.join(nc_dir, "*.nc"))
    # print(nc_files)
    for ncfile in nc_files:
        var_name = (os.path.splitext(os.path.basename(ncfile))[0]).split("_")[0]
        if os.path.exists(ncfile):
            # print(f"NetCDF file '{ncfile}' found. Checking last time step...")

            name, time_step, level = get_prod_names(var_name, FREQUENCY_SYMBOLS)
            try:
                # Open the existing file with dask enabled for lazy loading
                with xr.open_dataset(ncfile, chunks={"time": "auto"}) as existing_ds:
                    if "time" in existing_ds.coords:
                        last_timestep = existing_ds["time"].max().values
                        # Convert to datetime object for easier comparison
                        if np.issubdtype(last_timestep, np.datetime64):
                            last_timestep = pd.to_datetime(last_timestep)
                            var_ts[var_name] = last_timestep
                    existing_ds.close()
            except Exception as e:
                print(
                    f"Error opening existing NetCDF file: {e}. Assuming new file and starting from initial date."
                )

        else:
            print(f"NetCDF file '{ncfile}' not found.")
    return var_ts, time_step


def check_area_match(shp, nc_file):
    # Get shapefile bounding box
    minx, miny, maxx, maxy = shp.total_bounds  # (min_lon, min_lat, max_lon, max_lat)

    # Load NetCDF and extract lat/lon
    with xr.open_dataset(nc_file, chunks={"time": "auto"}) as ds:
        lats = ds["latitude"].values
        lons = ds["longitude"].values

        # If lat/lon are 2D (e.g., from satellite data), flatten
        if lats.ndim > 1:
            lats = lats.flatten()
        if lons.ndim > 1:
            lons = lons.flatten()

        # Check if all lat/lon values fall within bounding box
        all_within_lat = (lats >= miny).all() and (lats <= maxy).all()
        all_within_lon = (lons >= minx).all() and (lons <= maxx).all()

        if all_within_lat and all_within_lon:
            print("All lat/lon values fall within the shapefile bounding box.")
            ds.close()
            return True
        else:
            print("Some lat/lon values are outside the shapefile bounding box.")
            ds.close()
            return False


def setup_wapor_directories(project_foldr):
    """Creates project, nc, and csv directories."""
    nc_dir = os.path.join(project_foldr, "nc")
    csv_dir = os.path.join(project_foldr, "csvs")
    for d in [project_foldr, nc_dir, csv_dir]:
        os.makedirs(d, exist_ok=True)
    return nc_dir, csv_dir


def get_adjusted_period(variable, period, last_time_dict, shape, nc_dir, data_type):
    """Calculates the start date based on existing NetCDF files."""
    from pandas.tseries.offsets import DateOffset, MonthEnd

    if data_type != "raster" or not (
        isinstance(last_time_dict, dict) and variable in last_time_dict
    ):
        return period.copy(), get_yrs_mns_dekads(period)

    nc_path = os.path.join(nc_dir, f"{variable}.nc")
    if not check_area_match(shape, nc_path):
        print(f"Area mismatch for {variable}!")
        sys.exit()

    last_date = pd.to_datetime(last_time_dict[variable])
    if "-D" in variable:  # Decadal data (1–10, 11–20, 21–end)
        day = last_date.day

        if day <= 10:
            next_start = last_date.replace(day=11)
        elif day <= 20:
            next_start = last_date.replace(day=21)
        else:
            # move to first day of next month
            next_start = (last_date + MonthEnd(0)) + pd.Timedelta(days=1)

    elif "-M" in variable:  # Monthly data
        # move to first day of next month
        next_start = (last_date + MonthEnd(0)) + pd.Timedelta(days=1)

    else:  # Yearly data
        next_start = last_date + DateOffset(years=1)

    end_time = pd.to_datetime(period[1])
    if next_start > end_time:
        return [period[1], period[1]], None  # No new data

    new_period = [next_start.strftime("%Y-%m-%d"), period[1]]
    return new_period, get_yrs_mns_dekads(new_period)


def fetch_wapor_metadata(variable, period, shape, l3_code, time_info):
    """Retrieves product names, attributes, and download URLs."""
    name, frq, _ = get_prod_names(variable, FREQUENCY_SYMBOLS)

    # Handle Level 3 specifics
    if "L3-" in variable and l3_code is None:
        location, l3_code, _ = get_level3_code(shape)
        print("the AOI is within {0}".format(location))

    # Generate URLs
    if "AgERA" in variable:
        urls = get_agera5_ET0_PCP(variable, frq, time_info)
        scale_factor = 1
        unit = "mm"
    else:
        l3_param = l3_code if "L3-" in variable else None
        unit_ans_scale, urls = generate_urls_v3(variable, l3_param, period)
        scale_factor = unit_ans_scale[0][1]
        unit = unit_ans_scale[0][0]
    if not urls:
        return None, l3_code

    return {
        "name": name,
        "frq": frq,
        "urls": list(urls),
        "scale": scale_factor,
        "unit": unit,
    }, l3_code


def calculate_dekadal_days(time_rst):
    """calculate the number of days in the dekad from time steps with date such as "2024-02-01, 2024-02-11, 2024-02-21"."""
    days_idekad = []
    for dt in pd.DatetimeIndex(time_rst):
        if dt.day <= 10:
            days_idekad.append(10)
        elif dt.day <= 20:
            days_idekad.append(10)
        else:
            days_idekad.append(dt.days_in_month - 20)
    return days_idekad


def sample_url_worker(url, pts_coords):
    """Thread worker to sample points from a single URL."""
    try:
        with rasterio.open(url) as src:
            return [val[0] for val in src.sample(pts_coords)]
    except Exception:
        return [np.nan] * len(pts_coords)


def sample_url_worker_polygon(url, polygons, stat="mean"):
    """
    Robust polygon sampling with fallback for very small polygons.
    """
    results = []

    try:
        with rasterio.open(url) as src:
            res_x, res_y = src.res  # pixel size

            for poly in polygons:
                bounds = poly.bounds
                minx, miny, maxx, maxy = bounds

                # Check if polygon is smaller than a pixel
                if (maxx - minx) < res_x or (maxy - miny) < res_y:
                    # Use centroid pixel directly
                    row, col = src.index(poly.centroid.x, poly.centroid.y)
                    val = src.read(1)[row, col]
                    results.append(val)
                    continue

                # Normal windowed read
                window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
                window = window.round_offsets().round_lengths()

                # Ensure valid window
                if window.width <= 0 or window.height <= 0:
                    row, col = src.index(poly.centroid.x, poly.centroid.y)
                    val = src.read(1)[row, col]
                    results.append(val)
                    continue

                data = src.read(1, window=window, masked=True)
                transform = src.window_transform(window)

                mask = geometry_mask(
                    [poly],
                    transform=transform,
                    invert=True,
                    out_shape=data.shape,
                    # all_touched=True,
                )
                vals = data[mask]

                # If still empty, fallback to centroid
                if vals.size == 0:
                    row, col = src.index(poly.centroid.x, poly.centroid.y)
                    vals = np.array([src.read(1)[row, col]])

                # Compute statistic
                if stat == "mean":
                    results.append(np.nanmean(vals))
                elif stat == "sum":
                    results.append(np.nansum(vals))
                elif stat == "min":
                    results.append(np.nanmin(vals))
                elif stat == "max":
                    results.append(np.nanmax(vals))
                elif stat == "median":
                    results.append(np.nanmedian(vals))
                else:
                    raise ValueError("Unsupported statistic")

        return results

    except Exception as e:
        print(f"Sampling failed for {url}: {e}")
        return [np.nan] * len(polygons)


def process_raster_layers(
    meta, time_rst, shape, template, nc_dir, variable, xmin, ymin, xmax, ymax, crs
):
    """Downloads, clips, scales, and saves raster data to a NetCDF file."""
    nc_path = os.path.join(nc_dir, f"{variable}.nc")

    # Initialize NetCDF if it doesn't exist
    if not os.path.exists(nc_path):
        dims = {
            "time": None,
            "latitude": template.y.values,
            "longitude": template.x.values,
        }
        # Get attributes for the NetCDF variable
        _, _, level = get_prod_names(variable, FREQUENCY_SYMBOLS)
        # Note: get_attributes and init_nc are assumed to be defined in your environment
        # _, attrs = get_attributes(variable, meta["frq"], meta["name"], [], [], [])
        _, attrs = get_attributes(
            variable, meta["frq"], meta["name"], meta["unit"], [], [], []
        )
        if meta["frq"] == "dekadal" and "/day" in meta["unit"]:
            attrs.update({"unit": f"{attrs['unit'].rsplit('/', 1)[0]}/dekad"})
        var_spec = {meta["name"]: [("time", "latitude", "longitude"), attrs]}
        init_nc(nc_path, dims, var_spec, fill=np.nan, attr=attrs)

    # Process each timestep
    print(f"processing {variable} ...")
    # Handle dekadal 'per day' conversion
    days_in_dekad = calculate_dekadal_days(time_rst)
    for i, url in enumerate(meta["urls"]):
        date = time_rst[i]
        da = Open_with_rasterio(url, xmin, ymin, xmax, ymax)

        # Clip and scale
        ds_clipped = get_clipped_ds(da, shape, crs, template, meta["scale"])

        if meta["frq"] == "dekadal" and "/day" in meta["unit"]:
            ds_clipped = ds_clipped * days_in_dekad[i]

        fill_nc_one_timestep(nc_path, {meta["name"]: ds_clipped}, date)


def prepare_point_extraction(shape, points_col_name, first_url, crs, variable):
    """Filters shapefile points to those within the raster extent and returns coordinates."""
    print(f"processing {variable} ...")
    with rasterio.open(first_url) as src:
        raster_extent = box(*src.bounds)

    # Project shapefile to match raster CRS
    projected_gdf = shape.to_crs(crs)

    # Filter points that actually intersect the raster
    subset_gdf = projected_gdf[projected_gdf.geometry.intersects(raster_extent)].copy()

    if len(subset_gdf) == 0:
        print(f"None of the points are within the bounds of the {variable} raster.")
        return None, None

    # Handle naming/labeling
    if points_col_name in subset_gdf.columns.str.lower():
        column_label = subset_gdf.name.to_list()
    else:
        column_label = subset_gdf.index.to_list()

    # Use representative_point to handle both POINT and MULTIPOINT geometries
    subset_gdf["rep_point"] = subset_gdf.geometry.representative_point()
    pts_coords = [
        (x, y) for x, y in zip(subset_gdf.rep_point.x, subset_gdf.rep_point.y)
    ]

    print(f"{len(subset_gdf)} locations are within the bounds of {variable}")
    return pts_coords, column_label


def prepare_polygon_for_zonal_stat(shape, polygons_col_name, first_url, crs, variable):
    """Filters shapefile polygons to those within the raster extent and returns coordinates."""
    print(f"processing {variable} ...")
    with rasterio.open(first_url) as src:
        raster_extent = box(*src.bounds)

    # Project shapefile to match raster CRS
    projected_gdf = shape.to_crs(crs)

    # Filter points that actually intersect the raster
    subset_gdf = projected_gdf[projected_gdf.geometry.intersects(raster_extent)].copy()

    if len(subset_gdf) == 0:
        print(f"None of the polygons are within the bounds of the {variable} raster.")
        return None, None

    # Handle naming/labeling
    if polygons_col_name in subset_gdf.columns.str.lower():
        column_label = subset_gdf.block.to_list()
    else:
        column_label = subset_gdf.index.to_list()

    print(f"{len(subset_gdf)} polygons are within the bounds of {variable}")
    return subset_gdf, column_label


def align_lcc_raster(lcc_path, first_url):
    """
    Allocates memory ONLY for the area where LCC and Value raster overlap.
    The lcc raster is aligned to the raster from the url.
    """
    with rasterio.open(first_url) as ref:
        with rasterio.open(lcc_path) as src_lcc:
            # 1. Get the bounds of the LCC in the Value Raster's CRS
            # This 'crops' our focus to just the small LCC area
            lcc_bounds = src_lcc.bounds

            # 2. Create a window corresponding to these bounds in the ref raster
            window = from_bounds(*lcc_bounds, transform=ref.transform)
            window = window.round_offsets().round_lengths()

            # Check if the window is valid to avoid the 0x0 error
            if window.width <= 0 or window.height <= 0:
                raise ValueError(
                    "The LCC raster does not overlap with the reference raster's extent. Please check if the data to be downloded and the raster overlap."
                )

            # 3. Define the shape and transform of this specific window
            win_transform = ref.window_transform(window)
            win_shape = (int(window.height), int(window.width))

            # print(f"New windowed shape: {win_shape}")

            # 4. Allocate only for the window
            aligned_lcc_win = np.zeros(win_shape, dtype="uint8")

            reproject(
                source=rasterio.band(src_lcc, 1),
                destination=aligned_lcc_win,
                src_transform=src_lcc.transform,
                src_crs=src_lcc.crs,
                dst_transform=win_transform,
                dst_crs=ref.crs,
                resampling=Resampling.nearest,
                src_nodata=src_lcc.nodata,
                dst_nodata=0,
            )

    unique_types = np.unique(aligned_lcc_win)
    unique_types = unique_types[unique_types != 0]

    # We return the windowed array and the window metadata
    # so we know WHERE this slice belongs in the big raster later.
    return aligned_lcc_win, sorted(unique_types.tolist()), window


def align_lcc_raster_keep_resolution(lcc_path, first_url):
    """
    Reprojects the LCC raster to the CRS of first_url,
    but keeps the LCC's original resolution (e.g., 20m).
    """
    with rasterio.open(first_url) as ref:
        ref_crs = ref.crs

        with rasterio.open(lcc_path) as src_lcc:
            # 1. Calculate the transform/shape for the new CRS while keeping resolution
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_lcc.crs, ref_crs, src_lcc.width, src_lcc.height, *src_lcc.bounds
            )

            # 2. Allocate memory based on the new shape (keeping LCC resolution)
            win_shape = (int(dst_height), int(dst_width))
            # Check if the window is valid to avoid the 0x0 error
            if win_shape.width <= 0 or win_shape.height <= 0:
                raise ValueError(
                    "The LCC raster does not overlap with the reference raster's extent. Please check if the data to be downloded and the raster overlap."
                )

            aligned_lcc_win = np.zeros(win_shape, dtype="uint8")

            # 3. Perform the reprojection
            reproject(
                source=rasterio.band(src_lcc, 1),
                destination=aligned_lcc_win,
                src_transform=src_lcc.transform,
                src_crs=src_lcc.crs,
                dst_transform=dst_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest,  # Best for categorical data like LCC
                src_nodata=src_lcc.nodata,
                dst_nodata=0,
            )

            # 4. determine the geographic bounds
            west, south, east, north = rasterio.transform.array_bounds(
                dst_height, dst_width, dst_transform
            )
            lcc_bounds_in_ref_crs = (west, south, east, north)

    # Calculate unique classes for the stats loop
    unique_types = np.unique(aligned_lcc_win)
    unique_types = unique_types[unique_types != 0]

    # Note: We return lcc_bounds_in_ref_crs instead of a window
    return aligned_lcc_win, sorted(unique_types.tolist()), lcc_bounds_in_ref_crs


def sample_by_lcc_raster(value_url, aligned_lcc_win, lcc_columns, window, stat="mean"):
    """
    Uses the 'window' parameter to read only the matching slice of the Value Raster.
    The aligned_lcc_win is the lcc raster resample to match the value_url raster.
    """
    lcc_mask = aligned_lcc_win == 0
    results = []

    with rasterio.open(value_url) as src:
        # CRITICAL: Read only the window that matches our LCC slice
        v_data = src.read(1, window=window, masked=True)

        scale = src.scales[0] if src.scales[0] is not None else 1.0
        offset = src.offsets[0] if src.offsets[0] is not None else 0.0
        scr_nodata = src.nodata

        v_data = v_data.astype("float32")

        v_data = (v_data * scale) + offset
        # Basic check to ensure shapes match (rounding can sometimes be off by 1px)
        # We crop the value data to match the LCC array exactly
        v_data = v_data[: aligned_lcc_win.shape[0], : aligned_lcc_win.shape[1]]

        combined_mask = v_data.mask | lcc_mask
        v_flat = v_data.data[~combined_mask]
        l_flat = aligned_lcc_win[~combined_mask]

        for lcc_id in lcc_columns:
            group_vals = v_flat[l_flat == lcc_id]

            if group_vals.size > 0:
                val = (
                    np.nanmean(group_vals) if stat == "mean" else np.nanmax(group_vals)
                )
            else:
                val = np.nan
            results.append(val)

    return results


def sample_by_lcc_raster_keep_resolution(
    value_url, aligned_lcc_win, lcc_columns, lcc_bounds, stat="mean"
):
    """
    Resample the value_url to match the lcc resolution.
    value_url
    """
    target_height, target_width = aligned_lcc_win.shape

    with rasterio.open(value_url) as src:
        v_window = src.window(*lcc_bounds)
        # The url raster is resample to match the lcc raster
        v_data = src.read(
            1,
            window=v_window,
            out_shape=(target_height, target_width),
            resampling=Resampling.nearest,
            masked=True,
        )
        scale = src.scales[0] if src.scales[0] is not None else 1.0
        offset = src.offsets[0] if src.offsets[0] is not None else 0.0
        scr_nodata = src.nodata

    # 1. CAST TO FLOAT FIRST
    v_data_numeric = v_data.astype("float32")
    if scr_nodata is not None:
        # Replace the specific nodata value with NaN
        v_data_numeric = np.where(v_data_numeric == scr_nodata, np.nan, v_data_numeric)

    # scales is a list (one per band)
    v_data_numeric = (v_data_numeric * scale) + offset

    lcc_invalid = aligned_lcc_win == 0
    results = []

    for lcc_id in lcc_columns:
        # Create mask
        mask = (
            (aligned_lcc_win == lcc_id) & (~np.isnan(v_data_numeric)) & (~lcc_invalid)
        )
        group_vals = v_data_numeric[mask]

        if group_vals.size > 0:
            val = np.nanmean(group_vals) if stat == "mean" else np.nanmax(group_vals)
        else:
            val = np.nan
        results.append(val)
    return results
def get_raster_bbox(raster_path):
    with rasterio.open(raster_path) as src:
        return tuple(src.bounds)

def wapor_dl(region, variables, period, project_foldr, data_type="raster", **kwargs):
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=UserWarning)

    set_gdal_environment()
    client = initialize_multiprocessing()
    nc_dir, csv_dir = setup_wapor_directories(project_foldr)

    ##set datatype:
    if data_type == "zonal_stat":
        # Read user input if provided, else defaults
        stat = kwargs.get("stat", "mean")  # default mean
        polygons_col_name = kwargs.get("polygons_col_name", "block")
        print(f"Zonal stats on '{polygons_col_name}' using stat '{stat}'")
    elif data_type == "point":
        points_col_name = kwargs.get("points_col_name", "name")
        print(f"Sampling points using '{points_col_name}'")

    elif data_type == "raster":
        print("Processing raster")
    elif data_type == "sample_by_lcc":
        stat = kwargs.get("stat", "mean")  # default mean
        lcc_raster_path = kwargs.get("lcc_raster_path", "lcc_raster_path")
        use_lcc_resolution = kwargs.get("use_lcc_resolution", "no")
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    l3_code = None
    if data_type != "sample_by_lcc":
        shape, bbox = get_shape_and_bbox(data_type, region)
    else:
        bbox_rst = get_raster_bbox(lcc_raster_path)
        shape, bbox = get_shape_and_bbox(data_type, bbox_rst)
    last_time_dict, _ = check_exsting_netcdf_file(nc_dir)
    template = (
        get_template_rst(project_foldr, variables, shape, bbox)
        if data_type == "raster"
        else None
    )

    for variable in variables:
        # 1. Handle Time/Period logic
        current_period, time_info = get_adjusted_period(
            variable, period, last_time_dict, shape, nc_dir, data_type
        )
        # print("time_info", print(type(time_info)), time_info)
        if current_period[1] <= current_period[0]:
            print(f"No data to download for {variable} in this period.")
            continue

        # 2. Fetch URLs and Metadata
        meta, l3_code = fetch_wapor_metadata(
            variable, current_period, shape, l3_code, time_info
        )
        if not meta:
            print(f"Check WaPOR availability for {variable}. No URLs found.")
            sys.exit()

        # 3. Geo-referencing & Dekadal Setup
        if "AgERA" in variable:
            time_rst = time_index_for_AgERA(time_info, meta["frq"])
        else:
            time_rst = time_index_from_filenames(meta["frq"], meta["urls"])
        bbox2, crs = correct_bbox(variable, meta["urls"][0], bbox)
        xmin, ymin, xmax, ymax = bbox2

        days_idekad = (
            calculate_dekadal_days(time_rst) if meta["frq"] == "dekadal" else None
        )

        # 4. Processing logic
        if data_type == "raster":
            process_raster_layers(
                meta,
                time_rst,
                shape,
                template,
                nc_dir,
                variable,
                xmin,
                ymin,
                xmax,
                ymax,
                crs,
            )

        else:
            if data_type == "point":
                # --- START POINT EXTRACTION BLOCK ---

                # 1. Filter points and get coordinates
                pts_coords, column_labels = prepare_point_extraction(
                    shape, points_col_name, meta["urls"][0], crs, variable
                )

                if pts_coords is None:
                    continue  # Skip this variable if no points match

                # This maybe be faster
                with ThreadPoolExecutor(max_workers=10) as executor:
                    pt_data = list(
                        executor.map(
                            lambda u: sample_url_worker(u, pts_coords), meta["urls"]
                        )
                    )

                # 3. Create DataFrame and apply transformations
                df = pd.DataFrame(pt_data, columns=column_labels, index=time_rst)
                unit = meta["unit"]
                df = df * meta["scale"]

            elif data_type == "zonal_stat":  # Zonal stat or time series for an area
                shdf, column_labels = prepare_polygon_for_zonal_stat(
                    shape, polygons_col_name, meta["urls"][0], crs, variable
                )

                with ThreadPoolExecutor(max_workers=10) as executor:
                    poly_data = list(
                        executor.map(
                            lambda u: sample_url_worker_polygon(
                                u, shdf.geometry, stat="mean"
                            ),
                            meta["urls"],
                        )
                    )

                df = pd.DataFrame(poly_data, columns=column_labels, index=time_rst)
                unit = meta["unit"]
                df = df * meta["scale"]
            else:  ## extraction per llc
                # print("urls: ", meta["urls"])
                if use_lcc_resolution == "yes":
                    lcc_align_func = align_lcc_raster_keep_resolution
                    sample_func = sample_by_lcc_raster_keep_resolution
                else:
                    lcc_align_func = align_lcc_raster
                    sample_func = sample_by_lcc_raster

                lcc_raster, column_labels, window = lcc_align_func(
                    lcc_raster_path, meta["urls"][0]
                )

                with ThreadPoolExecutor(max_workers=10) as executor:
                    lcc_group_data = list(
                        executor.map(
                            lambda u: sample_func(
                                u, lcc_raster, column_labels, window, stat="mean"
                            ),
                            meta["urls"],
                        )
                    )

                df = pd.DataFrame(lcc_group_data, columns=column_labels, index=time_rst)
                unit = meta["unit"]
            # Apply dekadal days multiplier if frequency is dekadal
            if meta["frq"] == "dekadal":
                if "/day" in meta["unit"]:
                    unit = f"{meta['unit'].rsplit('/', 1)[0]}/dekad"
                else:
                    unit = meta["unit"]
                # Ensure days_idekad is aligned with the DataFrame rows

                if variable.startswith("L"):
                    df = df.mul(days_idekad, axis=0)

            # Apply scale factor and round
            #
            df = df.round(2)
            # 4. Save to CSV
            csv_path = os.path.join(
                csv_dir, f"{variable}_{unit.replace('/', '_per_')}.csv"
            )
            # print(csv_path)
            df.to_csv(csv_path)
            # --- END POINT EXTRACTION BLOCK ---

    client.close()


if __name__ == "__main__":
    # To chnage the working directory to the directory where this script is located
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    gdal.SetConfigOption(
        "GDAL_MEM_ENABLE_OPEN", "YES"
    )  # To allow Opening a MEM dataset with the MEM:::DATAPOINTER

    tt = time.time()

    # products = ["AgERA5-PCP-D"]
    products = ["L3-AETI-D", "L3-T-D", "L3-NPP-D", "L3-RSM-D", "L1-PCP-D", "L1-RET-D"]
    period = ["2023-10-01", "2024-04-30"]
    # products = ["L2-AETI-D"]
    # period = ["2024-05-01", "2024-10-31"]

    region = r"d:\modules\RS4AWM\2026\Gharbia_shp\Gharbia\Gharbia.shp"
    project_foldr = r"d:\modules\RS4AWM\2026"
    # region = r"d:\IPA\Mwea_blocks.json"
    # region = r"d:\modules\RS4AWM\2026\download_wapor_for_IPA\data\Erbil2.geojson"
    # project_foldr = r"d:\IPA"

    ### ---Select data type---
    # data_type = "point"
    data_type = "raster"
    # data_type = "zonal_stat"
    # data_type = "sample_by_lcc"

    if data_type == "sample_by_lcc":
        stat = "mean"  # possible stats inclyde max,
        lcc_raster_path = r"d:\modules\RS4AWM\2026\download_wapor_for_IPA\data\ESA_LC_2021_Erbil_20m.tif"
        wapor_dl(
            region,
            products,
            period,
            project_foldr,
            data_type=data_type,
            stat=stat,
            lcc_raster_path=lcc_raster_path,
            use_lcc_resolution="no",  # if yes, the data will be resampled to mach the lcc raster
        )

    if data_type == "zonal_stat":
        stat = "mean"  # possible stats inclyde max, min, meadian
        polygons_col_name = "block"
        wapor_dl(
            region,
            products,
            period,
            project_foldr,
            data_type=data_type,
            stat=stat,
            polygons_col_name=polygons_col_name,
        )
    if data_type == "point":
        points_col_name = "name"  ## The name of the points id
        wapor_dl(
            region,
            products,
            period,
            project_foldr,
            data_type=data_type,
            points_col_name=points_col_name,
        )
    if data_type == "raster":
        wapor_dl(region, products, period, project_foldr, data_type=data_type)

    elapsed = time.time() - tt
    print(
        ">> Time elapsed up to downloading the required data : "
        + "{0:.1f}".format(elapsed)
        + " s"
    )
