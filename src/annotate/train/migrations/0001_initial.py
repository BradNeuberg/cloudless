# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import jsonfield.fields


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Image',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('path', models.FilePathField(verbose_name=b'/Users/max/workspace/cloudless/src/annotate/train/static/training_images')),
                ('annotation', jsonfield.fields.JSONField(null=True, blank=True)),
            ],
        ),
    ]
