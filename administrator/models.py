from django.db import models

# Create your models here.

class RegisterTable(models.Model):
    Username = models.CharField(max_length=30, null=True, blank=True)
    Password = models.CharField(max_length=30,null=True, blank=True)
    Name=models.CharField(max_length=30,null=True,blank=True)
    Dob=models.DateField()
    
class CriminalTable(models.Model):
    Criminalname= models.CharField(max_length=30, null=True, blank=True)
    Image= models.FileField(upload_to='criminalimages',null=True, blank=True)
    Type=models.CharField(max_length=30,null=True,blank=True)
    Details=models.CharField(max_length=30,null=True,blank=True)
    Address=models.CharField(max_length=30,null=True,blank=True)

class FireTable(models.Model):
    Name= models.CharField(max_length=30, null=True, blank=True)
    Address= models.CharField(max_length=30,null=True, blank=True)
    Phone=models.CharField(max_length=30,null=True,blank=True)
    Email=models.CharField(max_length=30,null=True,blank=True)
    Location=models.CharField(max_length=30,null=True,blank=True)
    chat_id = models.CharField(max_length=50, null=True, blank=True)  # Telegram chat ID for notifications


class PoliceStation(models.Model):

    name = models.CharField(max_length=255,null=True,blank=True)

    location = models.CharField(max_length=255,null=True,blank=True)

    chat_id = models.CharField(max_length=50, null=True, blank=True)  # Telegram chat ID for notifications


class Camera(models.Model):

    camera_id=models.CharField(max_length=100,null=True,blank=True)

    cameraname = models.CharField(max_length=255,null=True,blank=True)

    location = models.CharField(max_length=255,null=True,blank=True)



    # ForeignKey relationships to FireStation and PoliceStation

    fire_station = models.ForeignKey(FireTable, null=True, blank=True, on_delete=models.CASCADE)

    police_station = models.ForeignKey(PoliceStation, null=True, blank=True, on_delete=models.CASCADE)



class Alert(models.Model):
    ALERT_TYPES = [
        ('INTRUSION', 'Intrusion'),
        ('FIRE', 'Fire'),
        ('OTHER', 'Other'),
    ]

    alert_type = models.CharField(max_length=50, choices=ALERT_TYPES)
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE)
    details = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.alert_type} - {self.details}"


