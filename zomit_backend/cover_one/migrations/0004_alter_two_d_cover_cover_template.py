# Generated by Django 5.0.7 on 2024-10-14 08:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('cover_one', '0003_remove_two_d_cover_updated_at_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='two_d_cover',
            name='cover_template',
            field=models.CharField(max_length=255),
        ),
    ]
