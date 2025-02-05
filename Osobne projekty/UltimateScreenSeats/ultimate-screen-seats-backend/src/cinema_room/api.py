from ninja_extra import Router

from .schemas import CinemaRoomSchema, CinemaRoomCreateSchema, CinemaRoomUpdateSchema
from core.schemas import MessageSchema
from .models import CinemaRoom

import helpers

router = Router()


@router.post('', response={201: CinemaRoomSchema, 400: MessageSchema, 500: MessageSchema}, auth=helpers.auth_required)
def create_cinema_room(request, payload: CinemaRoomCreateSchema):
    """Create new cinema room"""

    try:
        if CinemaRoom.objects.filter(name=payload.name).exists():
            return 400, {"message": "Cinema rooms with this name already exists."}
        
        cinema_room = CinemaRoom.objects.create(**payload.dict())

        return 201, cinema_room
    except Exception as e:
        return 500, {"message": "An unexpected error ocurred during creating cinema room."}
    

@router.get('', response={200: list[CinemaRoomSchema], 404: MessageSchema, 500: MessageSchema}, auth=helpers.auth_required)
def get_cienema_rooms(request):
    """Fetch list of cienema rooms"""

    try:
        cienema_rooms = CinemaRoom.objects.all()

        return 200, cienema_rooms
    except CinemaRoom.DoesNotExist:
        return 404, {"message": {"Cinema rooms not found."}}
    except Exception as e:
        return 500, {"message": "An unexpected error ocurred during fetching cinema rooms."}
    

@router.get('{cinema_room_id}', response={200: CinemaRoomSchema, 404: MessageSchema, 500: MessageSchema}, auth=helpers.auth_required)
def get_cienema_room(request, cinema_room_id):
    """Fetch cinema room by `cinema_room_id`"""

    try:
        cienema_room = CinemaRoom.objects.get(id=cinema_room_id)

        return 200, cienema_room
    except CinemaRoom.DoesNotExist:
        return 404, {"message": "Cinema room not found."}
    except Exception as e:
        return 500, {"message": "An unexpected error ocurred during fetching cinema room."}
    

@router.patch('{cinema_room_id}', response={200: CinemaRoomSchema, 404: MessageSchema, 500: MessageSchema}, auth=helpers.auth_required)
def update_cienema_rooms(request, payload: CinemaRoomUpdateSchema, cinema_room_id):
    """Update cinema room by `cinema_room_id`"""

    try:
        cienema_room = CinemaRoom.objects.get(id=cinema_room_id)

        for attr, value in payload.dict(exclude_unset=True).items():
            setattr(cienema_room, attr, value)

        cienema_room.save()

        return 200, cienema_room
    except CinemaRoom.DoesNotExist:
        return 404, {"message": "Cinema room not found."}
    except Exception as e:
        return 500, {"message": "An unexpected error ocurred during updating cinema room."}
    

@router.delete('{cinema_room_id}', response={200: MessageSchema, 404: MessageSchema, 500: MessageSchema}, auth=helpers.auth_required)
def delete_cienema_room(request, cinema_room_id):
    """Remove cinema room by `cinema_room_id`"""

    try:
        cienema_room = CinemaRoom.objects.get(id=cinema_room_id)

        cienema_room.delete()

        return 200, {"message": "Cinema room removed successfully."}
    except CinemaRoom.DoesNotExist:
        return 404, {"message": "Cinema room not found."}
    except Exception as e:
        return 500, {"message": "An unexpected error ocurred during removeing cinema room."}