# Generated by Django 3.1.3 on 2020-11-06 09:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('videos', '0002_auto_20201102_2044'),
    ]

    operations = [
        migrations.CreateModel(
            name='TemporaryVideoFile',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('video', models.FileField(upload_to='temp/videos/')),
            ],
        ),
    ]
