# Generated by Django 4.2.16 on 2024-10-28 06:45

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='CriminalTable',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Criminalname', models.CharField(blank=True, max_length=30, null=True)),
                ('Image', models.FileField(blank=True, null=True, upload_to='')),
                ('Type', models.CharField(blank=True, max_length=30, null=True)),
                ('Details', models.CharField(blank=True, max_length=30, null=True)),
                ('Address', models.CharField(blank=True, max_length=30, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='RegisterTable',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Username', models.CharField(blank=True, max_length=30, null=True)),
                ('Password', models.CharField(blank=True, max_length=30, null=True)),
                ('Name', models.CharField(blank=True, max_length=30, null=True)),
                ('Dob', models.DateField()),
            ],
        ),
    ]
