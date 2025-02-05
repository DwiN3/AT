import traceback
from typing import List, Optional
from ninja_extra import Router
from datetime import datetime
from django.utils import timezone
from django.utils.timezone import make_aware

from cinema_room.models import CinemaRoom
from core.schemas import MessageSchema
from movie.models import Movie
from reservation.models import Reservation

from .schemas import ShowingSchema, ShowingCreateSchema, ShowingSchemaList, ShowingUpdateSchema
from .models import Showing

import helpers

router = Router()


@router.post('', response={201: ShowingSchema, 404: MessageSchema, 500: MessageSchema}, auth=helpers.auth_required)
def create_showing(request, payload: ShowingCreateSchema):
    """Create a new showing"""

    try:
        movie = Movie.objects.get(id=payload.movie_id)
        cinema_room = CinemaRoom.objects.get(id=payload.cinema_room_id)

        showing = Showing.objects.create(
            movie=movie,
            cinema_room=cinema_room,
            date=payload.date,
            ticket_price=payload.ticket_price
        )

        return 201, showing
    except Movie.DoesNotExist:
        return 404, {"message": f"Movie with id {payload.movie_id} doesn't exist."}
    except CinemaRoom.DoesNotExist:
        return 404, {"message": f"Cinema room with id {payload.cinema_room_id} doesn't exist."}
    except Exception as e:
        traceback.print_exc()
        return 500, {"message": "An unexpected error ocurred during creating showing."}
    

@router.get('', response={200: list[ShowingSchema], 404: MessageSchema, 500: MessageSchema})
def get_showings(request,  limit: Optional[int] = None, movieId: Optional[int] = None):
    """Fetch list of schowings"""

    try:
        now = make_aware(datetime.now())

        showings = Showing.objects.filter(date__gt=now).order_by('date')


        if movieId is not None:
            showings = showings.filter(movie_id=movieId)

        if limit is not None:
            showings = showings[:limit]

        return 200, showings
    except Showing.DoesNotExist:
        return 404, {"message": f"Showings doesn't exist."}
    except Exception as e:
        return 500, {"message": "An unexpected error ocurred during fetching showings."}
    
    
@router.get('/list', response={200: List[ShowingSchemaList], 404: MessageSchema, 500: MessageSchema})
def get_showings(request, start_date: datetime = None, end_date: datetime = None):
    """Fetch showings with filtering by date range (default: from now to the end)"""

    try:
        if not start_date:
            start_date = timezone.now()

        if not end_date:
            end_date = timezone.make_aware(datetime(2100, 12, 31, 23, 59, 59))

        if timezone.is_naive(start_date):
            start_date = timezone.make_aware(start_date)

        if timezone.is_naive(end_date):
            end_date = timezone.make_aware(end_date)

        showings = Showing.objects.filter(
            date__gte=start_date,
            date__lte=end_date
        ).order_by('date')

        if not showings:
            return 404, {"message": "No showings found within the specified date range."}

        movie_showings = {}
        for showing in showings:
            movie_id = showing.movie.id
            if movie_id not in movie_showings:
                movie_showings[movie_id] = {
                    'movie': showing.movie,
                    'date_from': showing.date,
                    'date_to': showing.date,
                    'ticket_price': showing.ticket_price
                }
            else:
                movie_showings[movie_id]['date_from'] = min(movie_showings[movie_id]['date_from'], showing.date)
                movie_showings[movie_id]['date_to'] = max(movie_showings[movie_id]['date_to'], showing.date)

        result = []
        for movie_id, data in movie_showings.items():
            movie_data = data['movie']
            result.append({
                "id": movie_data.id,
                "movie": {
                    "id": movie_data.id,
                    "title": movie_data.title,
                    "image": movie_data.image,
                    "movie_length": movie_data.movie_length,
                    "age_classification": movie_data.age_classification
                },
                "cinema_room": None,
                "date_from": data['date_from'],
                "date_to": data['date_to'],
                "ticket_price": str(data['ticket_price'])
            })

        return 200, result

    except Exception as e:
        traceback.print_exc()
        return 500, {"message": "An unexpected error occurred while fetching the showings."}
    

@router.get('/{showing_id}', response={200: ShowingSchema, 404: MessageSchema, 500: MessageSchema}, auth=helpers.auth_required)
def get_showing(request, showing_id: int):
    """Fetch a single showing by `showing_id`"""
    try:
        showing = Showing.objects.get(id=showing_id)

        return 200, showing
    except Showing.DoesNotExist:
        return 404, {"message": f"Showing with id {showing_id} doesn't exist."}
    except Exception as e:
        return 500, {"message": "An unexpected error ocurred during fetching showings."}
    

@router.get('/{showing_id}/available-seats', response={200: list, 404: MessageSchema}, auth=helpers.auth_required)
def get_available_seats(request, showing_id: int):
    """
    Get available seats for a specific showing
    """
    try:
        showing = Showing.objects.select_related('cinema_room').get(id=showing_id)
        cinema_room = showing.cinema_room

        reserved_seats = Reservation.objects.filter(showing=showing).values_list('seat_row', 'seat_column')

        seat_layout = [row[:] for row in cinema_room.seat_layout]
        for seat_row, seat_column in reserved_seats:
            seat_layout[seat_row][seat_column] = 0

        available_seats = [
            {"row": row_index, "column": col_index}
            for row_index, row in enumerate(seat_layout)
            for col_index, seat in enumerate(row)
            if seat == 1
        ]

        return 200, available_seats
    except Showing.DoesNotExist:
        return 404, {"message": f"Showing with id {showing_id} doesn't exist."}


@router.patch('/{showing_id}', response={200: ShowingSchema, 404: MessageSchema, 500: MessageSchema}, auth=helpers.auth_required)
def update_showing(request, showing_id: int, payload: ShowingUpdateSchema):
    """Update an existing showing by `showing_id`."""

    try:
        showing = Showing.objects.get(id=showing_id)

        if payload.movie_id:
            try:
                movie = Movie.objects.get(id=payload.movie_id)
                showing.movie = movie
            except Movie.DoesNotExist:
                return 404, {"message": f"Movie with id {payload.movie_id} doesn't exist."}

        if payload.cinema_room_id:
            try:
                cinema_room = CinemaRoom.objects.get(id=payload.cinema_room_id)
                showing.cinema_room = cinema_room
            except CinemaRoom.DoesNotExist:
                return 404, {"message": f"Cinema room with id {payload.cinema_room_id} doesn't exist."}

        for attr, value in payload.dict(exclude_unset=True).items():
            if attr not in ("movie_id", "cinema_room_id"):
                setattr(showing, attr, value)

        showing.save()

        return 200, showing

    except Showing.DoesNotExist:
        return 404, {"message": f"Showing with id {showing_id} doesn't exist."}
    except Exception as e:
        return 500, {"message": "An unexpected error occurred during updating the showing."}


@router.delete('/{showing_id}', response={200: MessageSchema, 404: MessageSchema, 500: MessageSchema}, auth=helpers.auth_required)
def remove_showing(request, showing_id: int):
    """Remove a single showing by `showing_id`"""

    try:
        showing = Showing.objects.get(id=showing_id)

        showing.delete()

        return 200, {"message": f"Showing {showing_id} removed successfully."}
    except Showing.DoesNotExist:
        return 404, {"message": f"Showing with id {showing_id} doesn't exist."}
    except Exception as e:
        return 500, {"message": "An unexpected error ocurred during fetching showings."}
    