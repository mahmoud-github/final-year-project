# Generated by Django 4.2.5 on 2023-10-07 09:04

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('teacher', '0003_pdfdocument_course_pdfdocument_teacher'),
    ]

    operations = [
        migrations.DeleteModel(
            name='PDFDocument',
        ),
    ]