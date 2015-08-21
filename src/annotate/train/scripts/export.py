import argparse
import json
import os
import shutil

from django.apps import apps


def export(dirname):
    """
    Take all the annotated images, copy them to the specified directory,
    and write out annotated.json with the annotation for each image.
    """
    Image = apps.get_model('train', 'Image')
    annotated = Image.objects.filter(annotation__isnull=False)
    data = []
    for i in annotated:
        base = os.path.basename(i.path)
        # copy image to directory
        shutil.copy(i.path, os.path.join(dirname, base))
        # add bounding boxes to JSON
        data.append({
            'image_name': base,
            'image_annotation': i.annotation
        })

    with open(os.path.join(dirname, 'annotated.json'), 'w') as f:
        json.dump(data, f)

    return annotated.count()


def run(*args):
    parser = argparse.ArgumentParser(
        description='Export data JSON and annotated images to a directory'
    )
    parser.add_argument('dirname', help='Destination directory')
    args = parser.parse_args(args)
    num_exported = export(args.dirname)
    print '%s annotated images exported' % num_exported
