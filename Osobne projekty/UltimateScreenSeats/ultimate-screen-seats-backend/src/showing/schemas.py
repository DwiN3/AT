from ninja import Schema
from typing import Optional
from datetime import datetime
from decimal import Decimal
from pydantic import Field, field_validator

from cinema_room.schemas import CinemaRoomSchema
from movie.schemas import MovieSchema


class ShowingSchema(Schema):
    id: int
    movie: MovieSchema
    cinema_room: CinemaRoomSchema
    date: datetime
    ticket_price: Decimal

    class Config:
        from_attributes = True

class ShowingCreateSchema(Schema):
    movie_id: int
    cinema_room_id: int
    date: datetime = Field(..., description="The date and time of the showing.")
    ticket_price: Decimal = Field(..., description="The price of a ticket.")

    @field_validator("ticket_price")
    def validate_ticket_price(cls, value):
        return validate_ticket_price(value)
    
    def validate_date(value):
        if value <= datetime.now():
            raise ValueError("The date must be in the future.")
        return value
    
class ShowingUpdateSchema(Schema):
    movie_id: Optional[int] = None
    cinema_room_id: Optional[int] = None
    date: Optional[datetime] = None
    ticket_price: Optional[Decimal] = None

    @field_validator("ticket_price")
    def validate_ticket_price(cls, value):
        return validate_ticket_price(value)
    
    def validate_date(value):
        if value <= datetime.now():
            raise ValueError("The date must be in the future.")
        return value



def validate_ticket_price(value: Decimal):
    if value < 0 or value.as_tuple().exponent < -2:
        raise ValueError("Ticket price must be a positive value with up to 2 decimal places.")
    return value

def validate_date(value):
    if value <= datetime.now():
        raise ValueError("The date must be in the future.")
    return value


class MovieSchemaList(Schema):
    id: int
    title: str
    image: str
    movie_length: int
    age_classification: int

    class Config:
        orm_mode = True

class ShowingSchemaList(Schema):
    id: int
    date_from: datetime
    date_to: datetime
    movie: MovieSchemaList

    class Config:
        orm_mode = True