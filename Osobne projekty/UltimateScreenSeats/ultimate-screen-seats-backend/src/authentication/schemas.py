from ninja import Schema
from pydantic import EmailStr, Field, field_validator
from typing import Optional
import re


class RegisterSchema(Schema):
    username: str = Field(min_length=3, max_length=64)
    email: EmailStr
    password: str = Field(min_length=8)
    role: Optional[str] = "USER"

    @field_validator("password")
    def validate_password(cls, value):       
        return validate_password(value)
        
    @field_validator("role")
    def validate_role(cls, value):
        return validate_role(value)
    
class LoginSchema(Schema):
    email: EmailStr
    password: str

class UserDetailSchema(Schema):
    id: int
    username: str
    email: EmailStr
    role: str

class UserUpdateSchema(Schema):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    role: Optional[str] = None

    @field_validator("role")
    def validate_role(cls, value):
        return validate_role(value)

class ChangePasswordSchema(Schema):
    old_password: str
    new_password: str
    confirm_password: str

    @field_validator("new_password")
    def validate_new_password(cls, value):
        return validate_password(value)
    

# Validators
def validate_role(value):
    allowed_roles = ["USER", "ADMIN"]

    if value not in allowed_roles:
        raise ValueError(f"Invalid role: {value}")

    return value


def validate_password(value):       
    if not re.search(r'[A-Z]', value):
        raise ValueError("Password must contain at least one uppercase letter.")
    
    if not re.search(r'[a-z]', value):
        raise ValueError("Password must contain at least one lowercase letter.")
    
    if not re.search(r'[0-9]', value):
        raise ValueError("Password must contain at least one digit.")
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', value):
        raise ValueError("Password must contain at least one special character.")
    
    return value