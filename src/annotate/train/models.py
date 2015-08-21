import os

from django.conf import settings
from django.db import models

import jsonfield


class Image(models.Model):
    path = models.FilePathField(os.path.join(
        settings.BASE_DIR,
        'train/static/training_images'
    ))
    annotation = jsonfield.JSONField(blank=True, null=True)

    def url(self):
        url = self.path
        url = url.replace(settings.BASE_DIR, '').replace('/train/static/', '')
        url = settings.STATIC_URL + url
        return str(url)
