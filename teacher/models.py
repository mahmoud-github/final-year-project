from django.db import models
from django.contrib.auth.models import User
# from exam.models import Course

class Teacher(models.Model):
    user=models.OneToOneField(User,on_delete=models.CASCADE)
    profile_pic= models.ImageField(upload_to='profile_pic/Teacher/',null=True,blank=True)
    address = models.CharField(max_length=40)
    mobile = models.CharField(max_length=20,null=False)
    status= models.BooleanField(default=False)
    
    @property
    def get_name(self):
        return self.user.first_name+" "+self.user.last_name
    @property
    def get_instance(self):
        return self
    def __str__(self):
        return self.user.first_name
    

# class PDFDocument(models.Model):
#     title = models.CharField(max_length=255)
#     pdf_file = models.FileField(upload_to='pdfs/')
#     upload_date = models.DateTimeField(auto_now_add=True)
#     teacher =  models.ForeignKey(Teacher, on_delete=models.CASCADE , default= 1)
#     course = models.ForeignKey(Course, on_delete=models.CASCADE ,default=23)

#     def __str__(self):
#         return self.title