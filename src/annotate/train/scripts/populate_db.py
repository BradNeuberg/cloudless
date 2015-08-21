import argparse
import glob
import os
import shutil

from osgeo import gdal
from PIL import Image as pImage

from django.apps import apps
from django.conf import settings


def import_images(dirname, chunk_size=256):
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

    imported_ids = []
    for f in glob.glob(os.path.join(dirname, '*.tif')):
        print 'Processing %s' % f
        for chunk_f in chunk(f, chunk_size):
            converted_f = convert(chunk_f)
            os.remove(chunk_f)
            new_f = os.path.join(static_dir, os.path.basename(converted_f))
            shutil.move(converted_f, new_f)
            i = Image.objects.create(path=new_f)
            imported_ids.append(i.id)
    return imported_ids


def chunk(raster_filename, chunk_size=256, chunk_dir='/tmp/'):
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
            # TODO -make sure we don't have edge case issues
            chunk_filename = os.path.join(
                chunk_dir, '%s-%s-%s.tif' % (base, x, y)
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


def run(*args):
    parser = argparse.ArgumentParser(
        description='Populate database with images'
    )
    parser.add_argument('dirname', help='Directory containing raw geotiffs')
    parser.add_argument(
        'chunk_size', type=int, default=256,
        help='Desired pixel height/width of chunks'
    )
    args = parser.parse_args(args)
    ids = import_images(args.dirname, args.chunk_size)
    print '%s image chunks added to database' % len(ids)
