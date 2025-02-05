from django.db import models

class Genre(models.Model):
    name = models.CharField(max_length=64, unique=True)

    def __str__(self):
        return self.name


class Movie(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField()
    genre = models.ManyToManyField(Genre, related_name='movies')
    movie_length = models.IntegerField(default=0.0)
    age_classification = models.IntegerField(default=0)
    image = models.CharField(max_length=255)
    release_date = models.DateField(blank=True, null=True)
    background_image = models.CharField(max_length=255, default="")
    trailer_url = models.CharField(max_length=255, default="")
    cast = models.TextField(default="")
    director = models.CharField(max_length=255, default="")

    def __str__(self):
        return self.title
    

