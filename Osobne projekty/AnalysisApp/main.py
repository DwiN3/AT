import os
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from src.charts.context_sharing_class import ContextSharingVisualizer
from src.charts.patrol_distance_state import PatrolDistanceVisualizer
from src.charts.redundancy import RedundancyVisualizer
from src.charts.changing_state_details_class import ChangingStateDetailsVisualizer
from src.charts.distinct_details_class import DistinctDetailsVisualizer
from src.charts.firings_details_class import FiringDetailsVisualizer
from src.charts.first_patrol_data_class import PatrolDataVisualizer
from src.charts.ambulance_distance_and_time_class import AmbulanceDetailsVisualizer
from src.charts.comparison_of_ambulances_with_swat_class import AmbulancesAndSwatVisualizer
from src.charts.comparison_of_firings_and_interventions_class import InterventionAndFiringsVisualizer


chart_titles = [
    "I.1. Basic Information",
    "I.2. Incidents In Districts",
    "I.3. Changing State Details",
    "I.4. Patrols Distances",
    "I.6. Firings Details",
    "I.7. Interventions And Firings Details",
    "I.8. Comparison Of Ambulances With Swat",
    "Ambulance Details On firings",
    "II.1. Redundancy",
    "II.3. Context Sharing"
]

# Miasta
cities = [
    "",
    "Berlin",
    "Buenos Aires",
    "Chicago",
    "Krakow",
    "Tarnow",
    "Warszawa",
    "Wieden"
]

# Alert
def show_alert(text):
    messagebox.showinfo("Warning!", text)


###########################
######### WINDOW ##########
###########################
root = tk.Tk()
# root.geometry("1350x760") # 16:9
# root.geometry("1920x1080") # 16:9
root.state('zoomed') # Dopasuj do ekranu

# Ustalenie pełnej ścieżki do plików TCL
current_dir = os.path.dirname(os.path.realpath(__file__))
forest_light_path = os.path.join(current_dir, "src", "themes", "forest-light.tcl")
forest_dark_path = os.path.join(current_dir, "src", "themes", "forest-dark.tcl")

# Dodanie stylu
style = ttk.Style(root)
root.tk.call("source", forest_light_path)
root.tk.call("source", forest_dark_path)
style.theme_use("forest-dark")

# Ustawienie szerokości i wysokości komponentów
root.grid_columnconfigure(0, minsize=330)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0, weight=1)


###########################
####### NAVIGATION ########
###########################
frame1 = ttk.Frame(root, style="TFrame", height=root.winfo_height())
frame1.grid(row=0, column=0, sticky=tk.NSEW)
frame1.grid_columnconfigure(0, weight=1)
frame1.grid_propagate(False)

widgets_frame = ttk.LabelFrame(frame1, width=1, text="Set data")
widgets_frame.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")
widgets_frame.grid_columnconfigure(0, weight=1)

file_list_label = ttk.Label(widgets_frame, text="List of topics:")
file_list_label.grid(row=0, column=0, padx=15, pady=5, sticky="nsew")

# Wybór tematu wykresu
select_topic = ttk.Combobox(widgets_frame, values=chart_titles)
select_topic.current(0)
select_topic.grid(row=1, column=0, padx=15, pady=5, sticky="nsew")

city_frame = ttk.LabelFrame(widgets_frame, text="Set city/cities")
city_frame.grid(row=2, column=0, padx=20, pady=10)

# Wybór miast
city_comboboxes = []
selected_cities = set()

def show_available_cities(*args):
    selected_cities.clear()
    for combobox in city_comboboxes:
        selected_city = combobox.get()
        if selected_city:
            selected_cities.add(selected_city)

    available_cities = [city for city in cities if city not in selected_cities]

    for combobox in city_comboboxes:
        current_value = combobox.get()
        combobox['values'] = available_cities
        combobox.set('') if current_value in available_cities else current_value

for i in range(3):
    city_combobox = ttk.Combobox(city_frame, values=cities)
    city_combobox.grid(row=i, column=0, padx=15, pady=5, sticky="nsew")
    city_comboboxes.append(city_combobox)

for combobox in city_comboboxes:
    combobox.bind("<<ComboboxSelected>>", show_available_cities)
  
# Ładowanie danych
button = ttk.Button(widgets_frame, text="Load data", command=lambda: load_preset_options(select_topic.get()))
button.grid(row=3, column=0, padx=15, pady=5, sticky="nsew")


###########################
######### CONTENT #########
###########################
# Funkcja ładująca dane
def load_preset_options(chart_topic):
    if (len(selected_cities) == 0):
        show_alert("Choose at least 1 city!")
    else:
        # Zmniejszenie ramki
        children = frame1.winfo_children()
        for child in children:
            info = child.grid_info()
            if info['row'] >= 1:
                child.destroy()

        frame2 = ttk.Frame(root, height=root.winfo_height(), style="Green.TFrame")
        frame2.grid(row=0, column=1, sticky=tk.NSEW)

        match chart_topic:
            case "I.1. Basic Information":
                PatrolDataVisualizer(frame1, frame2, selected_cities)
            case "I.2. Incidents In Districts":
                DistinctDetailsVisualizer(frame1, frame2, selected_cities)
            case "I.3. Changing State Details":
                ChangingStateDetailsVisualizer(frame1, frame2, selected_cities)
            case "I.4. Patrols Distances":
                PatrolDistanceVisualizer(frame1, frame2, selected_cities)
            case "I.6. Firings Details":
                FiringDetailsVisualizer(frame1, frame2, selected_cities)
            case "Ambulance Details On firings":
                AmbulanceDetailsVisualizer(frame1, frame2, selected_cities)
            case "I.8. Comparison Of Ambulances With Swat":
                AmbulancesAndSwatVisualizer(frame1, frame2, selected_cities)
            case "I.7. Interventions And Firings Details":
                InterventionAndFiringsVisualizer(frame1, frame2, selected_cities)
            case "II.1. Redundancy":
                RedundancyVisualizer(frame1, frame2, selected_cities)
            case "II.3. Context Sharing":
                ContextSharingVisualizer(frame1, frame2, selected_cities)
            

###########################
########## STYLE ##########
###########################
style = ttk.Style(root)
root.title('Analysis Application')
style.configure("Green.TFrame", background="#217346")

root.mainloop()