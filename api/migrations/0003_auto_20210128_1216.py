# Generated by Django 3.1.5 on 2021-01-28 06:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0002_auto_20210127_2025'),
    ]

    operations = [
        migrations.AddField(
            model_name='data',
            name='gyro_x',
            field=models.FloatField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='data',
            name='gyro_y',
            field=models.FloatField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='data',
            name='gyro_z',
            field=models.FloatField(default=0),
            preserve_default=False,
        ),
    ]
