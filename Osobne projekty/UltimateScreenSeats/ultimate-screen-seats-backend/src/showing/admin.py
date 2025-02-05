from django.contrib import admin


from .models import Showing


class ShowingAdmin(admin.ModelAdmin):
    model = Showing
    list_display = ("id", "movie", "cinema_room", "date", "ticket_price")
    search_fields = ("id", "movie", "cinema_room", "date", "ticket_price")


admin.site.register(Showing, ShowingAdmin)