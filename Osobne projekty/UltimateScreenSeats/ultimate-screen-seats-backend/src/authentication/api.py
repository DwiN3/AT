from ninja import Router
from pydantic import ValidationError
from django.contrib.auth.hashers import make_password, check_password
from django.contrib.auth import authenticate
from ninja_jwt.tokens import RefreshToken

import helpers

from .models import User
from .schemas import ChangePasswordSchema, LoginSchema, RegisterSchema, UserDetailSchema, UserUpdateSchema
from core.schemas import MessageSchema

router = Router()

@router.post("/register", response= {201: UserDetailSchema, 400: MessageSchema})
def register(request, payload: RegisterSchema):
    """Create new account"""

    try:
        if User.objects.filter(email=payload.email).exists():
            return 400, {"message": "Email is already registered."}
        
        if User.objects.filter(username=payload.username).exists():
            return 400, {"message": "Username is already registered."}

        user_data = payload.dict()

        user_data['password'] = make_password(user_data['password'])

        user = User.objects.create(**user_data)

        return 201, user
    except ValidationError as e:
        return 400, {"message": str(e)}
    except Exception as e:
        return 400, {"message": "An unexpected error occurred."}
    

@router.post("/login", response={200: dict, 401: MessageSchema})
def login(request, payload: LoginSchema):
    """Log in on existing account"""

    user = authenticate(request, email=payload.email, password=payload.password)

    if user is None:
        return 401, {"message": "Invalid email or password"}
    
    refresh = RefreshToken.for_user(user)

    return {
        "refresh": str(refresh),
        "access": str(refresh.access_token),
        "username": user.username,
        "role": user.role
    }


@router.get("/user", response={200: UserDetailSchema, 400: MessageSchema}, auth=helpers.auth_required)
def get_user(request):
    """Get list of users"""

    try:
        user = request.user

        return 200, user
    except Exception as e:
        return 400, {"message": "An unexpected error occurred."}
    

@router.patch("/user", response={200: UserDetailSchema, 400: MessageSchema}, auth=helpers.auth_required)
def update_user(request, payload: UserUpdateSchema):
    """Update user credentials"""

    try:
        user = request.user

        if payload.email and User.objects.filter(email=payload.email).exclude(id=user.id).exists():
            return 400, {"message": "Email is already taken."}
        
        if payload.username and User.objects.filter(username=payload.username).exclude(id=user.id).exists():
            return 400, {"message": "Username is already taken."}

        for attr, value in payload.dict(exclude_unset=True).items():
            setattr(user, attr, value)

        user.save()

        return 200, user
    except ValidationError as e:
        return 400, {"message": str(e)}
    except Exception as e:
        return 400, {"message": "An unexpected error occurred."}
    

@router.post("/change-password", response={200: MessageSchema, 400: MessageSchema}, auth=helpers.auth_required)
def change_password(request, payload: ChangePasswordSchema):
    """Change password to account"""

    try:
        user = request.user
        if not check_password(payload.old_password, user.password):
            return 400, {"message": "Old password incorrect."}
        
        if payload.new_password != payload.confirm_password:
            return 400, {"message": "New passwords do not match."}
        
        user.password = make_password(payload.new_password)
        user.save()

        return 200, {"message": "Password changed successfully."}
    except ValidationError as e:
        return 400, {"message": str(e)}
    except Exception as e:
        return 400, {"message": "An unexpected error occurred."}