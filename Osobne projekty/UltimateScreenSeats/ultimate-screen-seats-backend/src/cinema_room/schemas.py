from typing import List, Optional
from ninja import Schema


class CinemaRoomSchema(Schema):
    id: int
    name: str
    seat_layout: List[List[int]]
    number_of_seats: int

    class Config:
        from_attributes = True

class CinemaRoomCreateSchema(Schema):
    name: str
    seat_layout: List[List[int]]


class CinemaRoomUpdateSchema(Schema):
    name: Optional[str] = None
    seat_layout: Optional[List[List[int]]] = None