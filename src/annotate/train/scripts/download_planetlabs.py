import argparse
import os
import requests

from osgeo import ogr, osr

SCENES_URL = "https://api.planet.com/v0/scenes/ortho/"

PLANET_KEY = os.environ.get('PLANET_KEY')
if not PLANET_KEY:
    raise Exception('PLANET_KEY environment variable not set!')


def download(lat, lng, buff_meters, download_dir='/tmp'):
    """
    Given a latitude, longitude, and a number of meters to buffer by,
    get all imagery that intersects the bounding box of the buffer and
    download it to the specified directory.  Return the names of the
    downloaded files.
    """
    pt = ogr.CreateGeometryFromWkt('POINT(%s %s)' % (lng, lat))
    pt = reproject(pt, 4326, 2163)  # from WGS84 to National Atlas
    buff = buffer_bbox(pt, buff_meters)
    buff = reproject(buff, 2163, 4326)

    data = requests.get(
        SCENES_URL,
        params={"intersects": buff.ExportToWkt()},
        auth=(PLANET_KEY, '')
    )

    downloaded_scenes = []
    for scene_data in data.json()["features"]:
        img_url = scene_data['properties']['links']['full']
        print 'Downloading image from %s' % img_url
        downloaded_scenes.append(download_image(img_url, download_dir))

    return downloaded_scenes


def reproject(geom, from_epsg, to_epsg):
    """
    Reproject the given geometry from the given EPSG code to another
    """
    source = osr.SpatialReference()
    source.ImportFromEPSG(from_epsg)

    target = osr.SpatialReference()
    target.ImportFromEPSG(to_epsg)

    transform = osr.CoordinateTransformation(source, target)

    geom.Transform(transform)

    return geom


def buffer_bbox(geom, buff):
    """
    Buffers the geom by buff and then calculates the bounding box.
    Returns a Geometry of the bounding box
    """
    b = geom.Buffer(buff)
    lng1, lng2, lat1, lat2 = b.GetEnvelope()
    wkt = """POLYGON((
        %s %s,
        %s %s,
        %s %s,
        %s %s,
        %s %s
    ))""" % (lng1, lat1, lng1, lat2, lng2, lat2, lng2, lat1, lng1, lat1)
    wkt = wkt.replace('\n', '')
    return ogr.CreateGeometryFromWkt(wkt)


def download_image(url, download_dir='/tmp'):
    """
    Downloads an image from a URL to the specified directory.

    Pulled from PlanetLabs docs with minor modifications.
    """
    r = requests.get(url, stream=True, auth=(PLANET_KEY, ''))
    if 'content-disposition' in r.headers:
        local_filename = r.headers['content-disposition'] \
            .split("filename=")[-1].strip("'\"")
    else:
        local_filename = '.'.join(url.split('/')[-2:])

    local_filename = os.path.join(download_dir, local_filename)

    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()

    return local_filename


def run(*args):
    parser = argparse.ArgumentParser(
        description='Download scenes from Planet'
    )
    parser.add_argument('lat', help='Latitude of interest')
    parser.add_argument('lng', help='Longitude of interest')
    parser.add_argument(
        'buffer', type=int, default=200,
        help='Meters to buffer lat/lng'
    )
    parser.add_argument(
        'dir', default='/tmp', help='Where to download files'
    )
    args = parser.parse_args(args)

    downloaded_scenes = download(
        args.lat, args.lng, args.buffer, args.dir
    )
    print '%s downloaded scenes' % len(downloaded_scenes)
