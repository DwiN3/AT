from django.db import models


class Showing(models.Model):
    movie = models.ForeignKey('movie.Movie', on_delete=models.CASCADE)
    cinema_room = models.ForeignKey('cinema_room.CinemaRoom', on_delete=models.CASCADE)
    date = models.DateTimeField()
    ticket_price = models.DecimalField(max_digits=6, decimal_places=2)

    def __str__(self):
        return f"{self.movie.title} - {self.date} in {self.cinema_room.name}"