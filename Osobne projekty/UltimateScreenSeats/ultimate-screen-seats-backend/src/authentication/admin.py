from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import User


class UserAdmin(UserAdmin):
    model = User
    list_display = ("id", "username", "email", "role")
    search_fields = ("id", "username", "email", "role")


admin.site.register(User, UserAdmin)