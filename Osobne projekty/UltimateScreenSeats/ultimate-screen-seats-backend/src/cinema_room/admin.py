from django.contrib import admin

from .models import CinemaRoom


class CinemaRoomAdmin(admin.ModelAdmin):
    model = CinemaRoom
    list_display = ("id", "name", "number_of_seats")
    search_fields = ("id", "name", "number_of_seats")


admin.site.register(CinemaRoom, CinemaRoomAdmin)