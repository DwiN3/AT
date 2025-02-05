from datetime import datetime
from typing import Optional
from ninja import Schema

from authentication.schemas import UserDetailSchema
from showing.schemas import ShowingSchema


class ReservationSchema(Schema):
    id: int
    user: UserDetailSchema
    showing: ShowingSchema
    seat_row: int
    seat_column: int
    reserve_at: datetime


class ReservationCreateSchema(Schema):
    showing_id: int
    seat_row: int
    seat_column: int


class ReservationUpdateSchema(Schema):
    showing_id: Optional[int] = None
    seat_row: Optional[int] = None
    seat_column: Optional[int] = None