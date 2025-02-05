from django.contrib import admin

from .models import Reservation


class ReservationAdmin(admin.ModelAdmin):
    model = Reservation
    list_display = ("id", "showing", "seat_row", "seat_column", "reserve_at")
    search_fields = ("id", "showing", "seat_row", "seat_column", "reserve_at")


admin.site.register(Reservation, ReservationAdmin)