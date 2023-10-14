from django.db import models

from student.models import Student
from teacher.models import Teacher 

class Course(models.Model):
   teacher =  models.ForeignKey(Teacher,on_delete=models.CASCADE)
   course_name = models.CharField(max_length=50)
   duration=models.PositiveIntegerField(default=30)
   def __str__(self):
        return self.course_name

class Question(models.Model):
    course=models.ForeignKey(Course,on_delete=models.CASCADE)
    marks=models.PositiveIntegerField(default=1)
    question=models.CharField(max_length=600)
    option1=models.CharField(max_length=200,null=True,blank=True)
    option2=models.CharField(max_length=200,null=True,blank=True)
    option3=models.CharField(max_length=200,null=True,blank=True)
    option4=models.CharField(max_length=200,null=True,blank=True)
    cat=(('Option1','Option1'),('Option2','Option2'),('Option3','Option3'),('Option4','Option4'))
    answer=models.CharField(max_length=200,choices=cat)
    rank=models.PositiveIntegerField(default=0,null=True,blank=True)
    def __str__(self):
        return self.question

class Result(models.Model):
    student = models.ForeignKey(Student,on_delete=models.CASCADE)
    exam = models.ForeignKey(Course,on_delete=models.CASCADE)
    marks = models.PositiveIntegerField()
    # grade = models.CharField(max_length=200,null=True,blank=True)
    date = models.DateTimeField(auto_now=True)



class PDFDocument(models.Model):
    title = models.CharField(max_length=255)
    pdf_file = models.FileField(upload_to='pdfs/')
    upload_date = models.DateTimeField(auto_now_add=True)
    teacher =  models.ForeignKey(Teacher, on_delete=models.CASCADE , default= 1)
    course = models.ForeignKey(Course, on_delete=models.CASCADE ,default=23)

    def __str__(self):
        return self.title