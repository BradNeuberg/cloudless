import glob
import os
import shutil

from osgeo import gdal
from PIL import Image as pImage

from django.apps import apps
from django.conf import settings


def import_images(dirname):
    """
    Given a directory with a bunch of raw geotiffs in it,
    cut them up into small pieces and load them into the database
    and the static folder
    """
    static_dir = os.path.join(
        settings.BASE_DIR,
        'train/static/training_images'
    )
    Image = apps.get_model('train', 'Image')
    for f in glob.glob(os.path.join(dirname, '*.tif')):
        for chunk_f in chunk(f):
            converted_f = convert(chunk_f)
            os.remove(chunk_f)
            new_f = os.path.join(static_dir, os.path.basename(converted_f))
            shutil.move(converted_f, new_f)
            Image.objects.create(path=new_f)


def chunk(raster_filename, chunk_size=244, crop_dir='/tmp/'):
    """
    Given a raster, a chunk size, and a directory to write into...
    Break the raster up into chunks of the appropriate size.
    """
    CROP_CMD = 'gdal_translate -srcwin %s %s %s %s %s %s'
    # % (xoff, yoff, xsize, ysize, src, dst)

    base = os.path.basename(os.path.splitext(raster_filename)[0])

    ds = gdal.Open(raster_filename)
    numPixelsWide, numPixelsHigh = ds.RasterXSize, ds.RasterYSize
    for x in range(0, numPixelsWide-chunk_size-1, chunk_size):
        for y in range(0, numPixelsHigh-chunk_size-1, chunk_size):
            chunk_filename = os.path.join(
                crop_dir, '%s-%s-%s.tif' % (base, x, y)
            )
            os.system(CROP_CMD % (
                x, y, chunk_size, chunk_size, raster_filename, chunk_filename
            ))
            yield chunk_filename


def convert(filename):
    """
    Converts a geotiff to a PNG in the same directory
    """
    new_f = os.path.join(
        os.path.dirname(filename),
        os.path.basename(filename).replace('.tif', '.png')
    )
    i = pImage.open(filename)
    i.save(new_f)
    return new_f
