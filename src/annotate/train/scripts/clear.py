import glob
import os

from django.apps import apps
from django.conf import settings


def run():
    Image = apps.get_model('train', 'Image')
    Image.objects.all().delete()
    files = glob.glob(os.path.join(
        settings.BASE_DIR, 'train', 'static', 'training_images', '*.png'
    ))
    for f in files:
        os.remove(f)
