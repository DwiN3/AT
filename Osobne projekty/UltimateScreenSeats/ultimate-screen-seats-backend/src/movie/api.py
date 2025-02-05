import traceback
from typing import Optional
from ninja_extra import Router

from core.schemas import MessageSchema
from .schemas import GenreCreateSchema, GenreSchema, MovieSchema, MovieCreateSchema, MovieUpdateSchema
from .models import Genre, Movie

import helpers

router = Router()

@router.post('', response={201: MovieSchema, 400: MessageSchema, 500: MessageSchema}, auth=helpers.auth_required)
def create_movie(request, payload: MovieCreateSchema):
    """Create a new movie"""

    try:
        genres = Genre.objects.filter(id__in=payload.genre_id)
        if not genres.exists():
            return 400, {"message": "Invalid genre IDs provided."}

        movie = Movie.objects.create(
            title=payload.title,
            description=payload.description,
            movie_length=payload.movie_length,
            age_classification=payload.age_classification,
            image=payload.image,
            release_date=payload.release_date,
            background_image=payload.background_image,
            trailer_url=payload.trailer_url,
            cast=payload.cast,
            director=payload.director
        )

        movie.genre.set(genres)
        return 201, movie
    except Exception as e:
        traceback.print_exc()
        return 500, {"message": "An unexpected error occurred during movie creation."}
 

@router.get('', response={200: list[MovieSchema], 404: MessageSchema, 500: MessageSchema})
def get_movies(request, limit: Optional[int] = None):
    """Fetch list of movies with an optional limit"""

    try:
        movies = Movie.objects.all().order_by('-release_date')

        if limit is not None:
            movies = movies[:limit]

        return 200, movies
    except Movie.DoesNotExist:
        return 404, {"message": "Movies not found."}
    except Exception as e:
        return 500, {"message": "An unexpected error occurred during fetching movies."}
    

@router.get('/genre', response={200: list[GenreSchema], 404: MessageSchema, 500: MessageSchema})
def get_genres(request):
    """Fetch list of genres"""

    try:
        genres = Genre.objects.all().order_by('name')
        return 200, genres
    except Genre.DoesNotExist:
        return 404, {"message": "Genres not found."}
    except Exception as e:
        return 500, {"message": "An unexpected error occurred while fetching genres."}


@router.get('{movie_id}', response={200: MovieSchema, 404: MessageSchema, 500: MessageSchema})
def get_movie(request, movie_id: int):
    """Fetch a single movie by `movie_id`"""

    try:
        movie = Movie.objects.get(id=movie_id)

        return 200, movie
    except Movie.DoesNotExist:
        return 404, {"message": f"Movie with id {movie_id} not found."}
    except Exception as e:
        traceback.print_exc()
        return 500, {"message": f"An unexpected error ocurred during fetching movie with id {movie_id}."}
    

@router.patch('{movie_id}', response={200: MovieSchema, 400: MessageSchema, 404: MessageSchema, 500: MessageSchema}, auth=helpers.auth_required)
def update_movie(request, payload: MovieUpdateSchema, movie_id: int):
    """Update an existing movie by `movie_id`"""

    try:
        movie = Movie.objects.get(id=movie_id)

        if payload.genre_id is not None:
            genres = Genre.objects.filter(id__in=payload.genre_id)
            if not genres.exists():
                return 400, {"message": "Invalid genre IDs provided."}
            movie.genre.set(genres)

        for attr, value in payload.dict(exclude_unset=True).items():
            if attr != "genre_ids":
                setattr(movie, attr, value)

        movie.save()
        return 200, movie
    except Movie.DoesNotExist:
        return 404, {"message": f"Movie with id {movie_id} not found."}
    except Exception as e:
        traceback.print_exc()
        return 500, {"message": "An unexpected error occurred during movie update."}


@router.delete('{movie_id}', response={200: MessageSchema, 404: MessageSchema, 500: MessageSchema}, auth=helpers.auth_required)
def delete_movie(request, movie_id: int):
    "Remove a movie by `movie_id`"
    
    try:
        movie = Movie.objects.get(id=movie_id)

        movie.delete()

        return 200, {"message": f"Movie {movie_id} removed successfully."}
    except Movie.DoesNotExist:
        return 404, {"message": f"Movie with id {movie_id} not found."}
    except Exception as e:
        return 500, {"message": f"An unexpected error ocurred during removeing movie with id {movie_id}."}
    

@router.post('/genre', response={201: GenreSchema, 500: MessageSchema}, auth=helpers.auth_required)
def create_genre(request, payload: GenreCreateSchema):
    """Create a new genre"""

    try:
        genre = Genre.objects.create(**payload.dict())
        return 201, genre
    except Exception as e:
        return 500, {"message": "An unexpected error occurred while creating the genre."}


@router.get('/genre/{genre_id}', response={200: GenreSchema, 404: MessageSchema, 500: MessageSchema})
def get_genre(request, genre_id: int):
    """Fetch details of a specific genre"""

    try:
        genre = Genre.objects.get(id=genre_id)
        return 200, genre
    except Genre.DoesNotExist:
        return 404, {"message": "Genre not found."}
    except Exception as e:
        return 500, {"message": "An unexpected error occurred while fetching the genre."}


@router.put('/genre/{genre_id}', response={200: GenreSchema, 404: MessageSchema, 500: MessageSchema}, auth=helpers.auth_required)
def update_genre(request, genre_id: int, payload: GenreCreateSchema):
    """Update a specific genre"""

    try:
        genre = Genre.objects.get(id=genre_id)
        for attr, value in payload.dict().items():
            setattr(genre, attr, value)
        genre.save()
        return 200, genre
    except Genre.DoesNotExist:
        return 404, {"message": "Genre not found."}
    except Exception as e:
        return 500, {"message": "An unexpected error occurred while updating the genre."}


@router.delete('/genre/{genre_id}', response={204: None, 404: MessageSchema, 500: MessageSchema}, auth=helpers.auth_required)
def delete_genre(request, genre_id: int):
    """Delete a specific genre"""

    try:
        genre = Genre.objects.get(id=genre_id)
        genre.delete()
        return 204, None
    except Genre.DoesNotExist:
        return 404, {"message": "Genre not found."}
    except Exception as e:
        return 500, {"message": "An unexpected error occurred while deleting the genre."}