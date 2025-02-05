from django.db import models


class Reservation(models.Model):
    showing = models.ForeignKey('showing.Showing', on_delete=models.CASCADE)
    user = models.ForeignKey('authentication.User', on_delete=models.CASCADE)
    seat_row = models.IntegerField()
    seat_column = models.IntegerField()
    reserve_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Reservation for Showing {self.showing.id}, Seat ({self.seat_row}, {self.seat_column})"