# Generated by Django 3.1.3 on 2020-11-19 11:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('videos', '0006_videoprocessingmonitor_has_slices'),
    ]

    operations = [
        migrations.AddField(
            model_name='videoprocessingmonitor',
            name='events',
            field=models.TextField(default=''),
        ),
    ]
