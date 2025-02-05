from django.db import models

class CinemaRoom(models.Model):
    name = models.CharField(max_length=255, unique=True)
    seat_layout = models.JSONField()

    @property
    def number_of_seats(self):
        return sum(sum(1 for seat in row if seat != -1) for row in self.seat_layout)
    
    def __str__(self):
        return self.name