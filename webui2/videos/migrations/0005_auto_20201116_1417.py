# Generated by Django 3.1.3 on 2020-11-16 14:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('videos', '0004_auto_20201106_1044'),
    ]

    operations = [
        migrations.AlterField(
            model_name='videoprocessingmonitor',
            name='status',
            field=models.CharField(choices=[('initial', 'Start'), ('uploading', 'Uploading'), ('uploaded', 'Uploaded'), ('processing', 'Processing'), ('done', 'Done')], default='initial', max_length=10),
        ),
    ]
