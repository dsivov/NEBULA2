# Generated by Django 3.2.4 on 2021-07-09 09:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('videos', '0009_alter_temporaryvideofile_video'),
    ]

    operations = [
        migrations.CreateModel(
            name='UserSetting',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user', models.CharField(max_length=100)),
                ('settings', models.CharField(max_length=25)),
            ],
        ),
    ]
