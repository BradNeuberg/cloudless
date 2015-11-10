import argparse
import os
import sys
import requests
import shutil
from urlparse import urlparse

from osgeo import ogr, osr

PLANET_KEY = os.environ.get('PLANET_KEY')
if not PLANET_KEY:
    raise Exception('PLANET_KEY environment variable not set!')


def download(lat, lng, buff_meters, download_dir='/tmp', image_type='planetlabs'):
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

    if image_type == 'planetlabs':
        scenes_url = "https://api.planet.com/v0/scenes/ortho/"
    elif image_type == 'rapideye':
        scenes_url = "https://api.planet.com/v0/scenes/rapideye/"

    # Download the initial scenes URL and also any paged results that come after that.
    downloaded_scenes = []
    next_url = scenes_url
    params = {"intersects": buff.ExportToWkt()}
    while next_url != None:
        next_url = download_results(next_url, params, downloaded_scenes, download_dir, image_type)
        print "\nWorking with next page of results: %s" % next_url

    return downloaded_scenes

def download_results(results_url, params, downloaded_scenes, download_dir, image_type):
    """
    Actually downloads from the given URL; if we are dealing with multiple pages of results
    this could be a URL for the next page of image results.
    """
    data = requests.get(
        results_url,
        params,
        auth=(PLANET_KEY, '')
    )

    for scene_data in data.json()["features"]:
        if image_type == 'planetlabs':
            # Planet Labs ortho satellite
            img_url = scene_data['properties']['links']['full']
        elif image_type == 'rapideye':
            # Rapideye
            img_url = scene_data["properties"]["data"]["products"]["analytic"]["full"]

        if does_download_exist(img_url, download_dir) == True:
            print '\nAlready downloaded %s' % img_url
            continue

        # Sometimes the downloaded image can be corrupted - keep trying until we have
        # an image we can work with.
        finished = False
        while not finished:
            try:
                print '\nDownloading image from %s' % img_url
                image_filename = download_image(img_url, download_dir)

                if image_type == 'rapideye':
                    # RapidEye images are 16-bit non-color corrected images by default.
                    image_filename = from_analytic_to_visual(image_filename, download_dir)

                finished = True
            except:
                print "Unexpected error dealing with image:", sys.exc_info()[0]

        downloaded_scenes.append(image_filename)

    return data.json()["links"]["next"]

def from_analytic_to_visual(analytic_filename, download_dir='/tmp'):
    """
    RapidEye images are 16-bit non-color corrected images. We need to extract the RGB bands and
    shift the image visually to be more friendly for people (they are fairly dark by default).
    Process adapted from: https://www.mapbox.com/blog/processing-rapideye-imagery/
    """
    GDAL_TRANSLATE = 'gdal_translate -b 3 -b 2 -b 1 %s %s'
    # % (input, output)

    GDAL_WARP = 'gdalwarp -co photometric=RGB -co tfw=yes -t_srs EPSG:3857 %s %s'
    # % (input, output)

    CONVERT = 'convert -sigmoidal-contrast 20x5%% -depth 8 %s %s'
    # % (input, output)

    base = os.path.basename(os.path.splitext(analytic_filename)[0])

    # Extract the RGB bands.
    print "Extracting RGB bands..."
    translate_filename = os.path.join(
        download_dir, '%s-rgb.tif' % (base)
    )
    os.system(GDAL_TRANSLATE % (
        analytic_filename, translate_filename
    ))

    # Project the image.
    print "Projecting image..."
    proj_filename = os.path.join(
        download_dir, '%s-proj.tif' % (base)
    )
    os.system(GDAL_WARP % (
        translate_filename, proj_filename
    ))

    # Convert it to a color scheme that is friendlier for people.
    print "Converting to visual color scheme..."
    bright_filename = os.path.join(
        download_dir, '%s-bright.tif' % base
    )
    os.system(CONVERT % (
        proj_filename, bright_filename
    ))

    # Cleanup
    print "Cleaning up..."
    shutil.move(bright_filename, analytic_filename)
    os.remove(translate_filename)
    os.remove(proj_filename)
    os.remove(os.path.join(download_dir, '%s-proj.tfw' % (base)))
    os.remove(os.path.join(download_dir, '%s-rgb.tif.aux.xml' % (base)))

def reproject(geom, from_epsg, to_epsg):
    """
    Reproject the given geometry from the given EPSG code to another
    """
    # Note: this is currently only accurate for the U.S.
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
    """
    local_filename = get_download_filename(url, download_dir)

    print "Saving raw downloaded image to %s..." % local_filename
    r = requests.get(url, stream=True, auth=(PLANET_KEY, ''))
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()

    return local_filename

def does_download_exist(url, download_dir='/tmp'):
    """
    Whether we've already downloaded this URL.
    """
    local_filename = get_download_filename(url, download_dir)
    return os.path.exists(local_filename)

def get_download_filename(url, download_dir='/tmp'):
    """
    Figures out what the filename should be for a downloaded URL.
    """
    url = urlparse(url)
    local_filename = url.path.split("/")[-2] + ".tif"
    local_filename = os.path.join(download_dir, local_filename)
    return local_filename

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download scenes from Planet'
    )
    parser.add_argument('lat', help='Latitude of interest')
    parser.add_argument('lng', help='Longitude of interest')
    parser.add_argument(
        '--buffer', type=int, default=200,
        help='Meters to buffer lat/lng'
    )
    parser.add_argument(
        '--dir', default='/tmp', help='Where to download files'
    )
    parser.add_argument(
        '--image_type', default='planetlabs', help="""Whether to download 'planetlabs', 'rapideye', or 'both'
        image data sets for the given lat/lng""")
    args = parser.parse_args()

    image_type = args.image_type
    if image_type == 'both':
        planetlab_scenes = download(args.lat, args.lng, args.buffer, args.dir, 'planetlabs')
        print '%s downloaded PlanetLab scenes' % len(planetlab_scenes)

        rapideye_scenes = download(args.lat, args.lng, args.buffer, args.dir, 'rapideye')
        print '%s downloaded RapidEye scenes' % len(rapideye_scenes)
    else:
        downloaded_scenes = download(args.lat, args.lng, args.buffer, args.dir, args.image_type)
        print '%s downloaded %s scenes' % (len(downloaded_scenes), args.image_type)
