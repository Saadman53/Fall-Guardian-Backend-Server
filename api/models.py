from django.db import models

class Data(models.Model):
	timestamp = models.IntegerField()
	acc_x = models.FloatField()
	acc_y = models.FloatField()
	acc_z = models.FloatField()
	gyro_x = models.FloatField()
	gyro_y = models.FloatField()
	gyro_z = models.FloatField()

	def __str__(self):
		return str(self.timestamp)


