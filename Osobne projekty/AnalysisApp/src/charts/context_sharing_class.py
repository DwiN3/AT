import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
from src.dataOperations.export_chart import export_plot_to_image
from matplotlib.ticker import MultipleLocator
from src.dataOperations.export_data import export_to_csv
from src.dataOperations.load_data import load_data


class ContextSharingVisualizer():
    def __init__(self, frame1, frame2, selected_cities):
        super().__init__()

        self.frame1 = frame1
        self.frame2 = frame2

        chart_topic = "Changing State Details"

        self.data = load_data(selected_cities, chart_topic)

        self.selected_cities = list(selected_cities)

        ###########################
        ######### ZMIENNE #########
        ###########################
        self.city_var = tk.StringVar()

        self.district_dropdown_var = tk.StringVar()
        self.district_level_dropdown_var = tk.StringVar()
        self.state_var = tk.StringVar()
        self.time_unit_var = tk.StringVar()

        self.start_time_hour_var = tk.StringVar()
        self.start_time_minute_var = tk.StringVar()
        self.end_time_hour_var = tk.StringVar()
        self.end_time_minute_var = tk.StringVar()

        self.state_radios = []
        self.city_radios = []
        self.city_state_radios = []
        self.presentation_mode = 'chart'

        self.states = [
            "All",
            "PATROLLING",
            "FIRING",
            "INTERVENTION",
            "NEUTRALIZED",
            "CALCULATING_PATH",
            "RETURNING_TO_HQ",
            "TRANSFER_TO_INTERVENTION",
            "TRANSFER_TO_FIRING"
        ]

        self.safety_levels = [
            "All",
            "Safe",
            "NotSafe",
            "RatherSafe"
        ]

        self.time_units = [
            "Hour",
            "Minute",
            "Second"
        ]

        self.district_name = self.data['districtName'].unique()

        ###########################
        ########### OKNA ##########
        ###########################
        # Nawigacja
        self.options_frame = ttk.LabelFrame(self.frame1, text="Set options")
        self.options_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.options_frame.columnconfigure(0, weight=1)

        # Export
        self.export_frame = ttk.LabelFrame(self.frame1, text="Export")
        self.export_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        self.export_frame.columnconfigure(0, weight=1)

        # Panel Wykresu
        self.frame2.grid_columnconfigure(0, weight=1)
        self.frame2.grid_rowconfigure(0, weight=3)

        self.frame_chart = ttk.Frame(self.frame2, style="TNotebook", padding=10)
        self.frame_chart.grid(row=0, column=0, sticky=tk.NSEW)
        self.frame_chart.grid_propagate(False)

        # Panel Analizy
        self.frame2.grid_columnconfigure(0, weight=1)
        self.frame2.grid_rowconfigure(1, weight=2)

        self.frame_analise = ttk.Frame(self.frame2, style="TNotebook", padding=10)
        self.frame_analise.grid(row=1, column=0, sticky=tk.NSEW)
        self.frame_analise.grid_propagate(False)

        # Wywołanie tworznie panelu nawigacyjnego
        self.create_nawigation_panel()


    # Tworzenie nawigacji
    def create_nawigation_panel(self):
        # Wyczyszczenie frame
        for widget in self.options_frame.winfo_children():
            widget.destroy()
    
        set_city_frame = ttk.LabelFrame(self.options_frame, text="Set city")
        set_city_frame.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")

        self.selected_cities = ["All"] + self.selected_cities
        city_dropdown = ttk.Combobox(set_city_frame, values=self.selected_cities, width=28, textvariable=self.city_var)
        city_dropdown.current(0)
        city_dropdown.bind("<<ComboboxSelected>>", self.prepare_data)
        city_dropdown.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        set_time_unit_frame = ttk.LabelFrame(self.options_frame, text="Set Time Unit")
        set_time_unit_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        safety_level_dropdown = ttk.Combobox(set_time_unit_frame, values=self.time_units, width=28, textvariable=self.time_unit_var)
        safety_level_dropdown.current(1)
        safety_level_dropdown.bind("<<ComboboxSelected>>", self.prepare_data)
        safety_level_dropdown.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        set_safety_level_frame = ttk.LabelFrame(self.options_frame, text="Set District | Safety Level")
        set_safety_level_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")

        mode_switch = ttk.Checkbutton(set_safety_level_frame, text="Set District | Safety Level", style="Switch", command=lambda: self.toggle_mode(mode_switch, set_safety_level_frame))
        mode_switch.grid(row=0, column=0, padx=5, pady=10, sticky="nsew")
        
        set_state_frame = ttk.LabelFrame(self.options_frame, text="Set state")
        set_state_frame.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")

        state_dropdown = ttk.Combobox(set_state_frame, values=self.states, width=28, textvariable=self.state_var)
        state_dropdown.current(0)
        state_dropdown.bind("<<ComboboxSelected>>", self.prepare_data)
        state_dropdown.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        set_time_frame = ttk.LabelFrame(self.options_frame, text="Set Start Time")
        set_time_frame.grid(row=4, column=0, padx=20, pady=10, sticky="nsew")

        set_start_time_frame = ttk.LabelFrame(set_time_frame, text="Set Start Time")
        set_start_time_frame.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")

        self.start_time_hour_var.set("12")
        spin_box = ttk.Spinbox(set_start_time_frame, from_=0, to=23, textvariable=self.start_time_hour_var, wrap=True, width=4)
        spin_box.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.start_time_minute_var.set("30")
        spin_box = ttk.Spinbox(set_start_time_frame, from_=0, to=59, textvariable=self.start_time_minute_var, wrap=True, width=4)
        spin_box.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        set_end_time_frame = ttk.LabelFrame(set_time_frame, text="Set End Time")
        set_end_time_frame.grid(row=1, column=0, padx=20, pady=0, sticky="nsew")

        self.end_time_hour_var.set("14")
        spin_box = ttk.Spinbox(set_end_time_frame, from_=0, to=23, textvariable=self.end_time_hour_var, wrap=True, width=4)
        spin_box.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.end_time_minute_var.set("30")
        spin_box = ttk.Spinbox(set_end_time_frame, from_=0, to=59, textvariable=self.end_time_minute_var, wrap=True, width=4)
        spin_box.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        button = ttk.Button(set_time_frame, text="Set time", width=16, command=self.prepare_data)
        button.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")

        # display_frame = ttk.LabelFrame(self.options_frame, text="Display")
        # display_frame.grid(row=5, column=0, padx=20, pady=10, sticky="nsew")

        # mode_switch_presentation = ttk.Checkbutton(display_frame, text="Chart | Table", style="Switch", command=lambda: self.toggle_mode_presentation(mode_switch_presentation))
        # mode_switch_presentation.grid(row=0, column=0, padx=5, pady=10, sticky="nsew")

        self.toggle_mode(mode_switch, set_safety_level_frame)
    
    # Obsługa Switch'a
    def toggle_mode(self, mode_switch, frame):
        for radio in self.city_state_radios:
            radio.destroy()

        if mode_switch.instate(["selected"]):
            self.district_level_dropdown_var.set("All")

            if self.city_var.get() == 'All':
                all_district_name = self.data['districtName'].unique()
            else:
                city_data = self.data[self.data['City'] == self.city_var.get()]
                all_district_name = city_data['districtName'].unique()

            district_name_numbers = list(all_district_name)
            district_name_numbers.insert(0, "All")

            district_dropdown = ttk.Combobox(frame, values=district_name_numbers, width=28, textvariable=self.district_dropdown_var)
            district_dropdown.current(0)
            district_dropdown.bind("<<ComboboxSelected>>", self.prepare_data)
            district_dropdown.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        else:
            self.district_dropdown_var.set("All")

            safety_level_dropdown = ttk.Combobox(frame, values=self.safety_levels, width=28, textvariable=self.district_level_dropdown_var)
            safety_level_dropdown.current(0)
            safety_level_dropdown.bind("<<ComboboxSelected>>", self.prepare_data)
            safety_level_dropdown.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.prepare_data()


    # Obsługa Switch'a
    def toggle_mode_presentation(self, mode_switch):
        for widget in self.frame_chart.winfo_children():
            widget.destroy()

        if mode_switch.instate(["selected"]):
            self.presentation_mode = 'table'
        else:
            self.presentation_mode = 'chart'
        
        self.prepare_data()


    # Przygotowanie danych
    def prepare_data(self, event=None):
        # Tworzenie kopii danych do filtrowania
        filtered_df = self.data.copy() 

        if self.city_var.get() != "All":
            filtered_df = filtered_df[filtered_df["City"] == self.city_var.get()]

        filtered_df['simulationTime'] = pd.to_datetime(filtered_df['simulationTime[s]'], unit='s')
        filtered_df['Hour'] = filtered_df['simulationTime'].dt.hour
        filtered_df['Minute'] = filtered_df['simulationTime'].dt.minute

        self.start_time = f"{self.start_time_hour_var.get()}:{self.start_time_minute_var.get()}"
        self.end_time = f"{self.end_time_hour_var.get()}:{self.end_time_minute_var.get()}"

        # Konwersja godziny początkowej i końcowej na format datetime
        start_datetime = pd.to_datetime(self.start_time, format='%H:%M').time()
        end_datetime = pd.to_datetime(self.end_time, format='%H:%M').time()

        # Filtrowanie danych na podstawie przedziału czasowego
        filtered_df = filtered_df[(filtered_df['simulationTime'].dt.time >= start_datetime) & (filtered_df['simulationTime'].dt.time <= end_datetime)]

        if self.district_dropdown_var.get() != "All":
            filtered_df = filtered_df[filtered_df['districtName'] == self.district_dropdown_var.get()]

        if self.district_level_dropdown_var.get() != "All":
            filtered_df = filtered_df[filtered_df['districtSafetyLevel'] == self.district_level_dropdown_var.get()]

        if self.state_var.get() != "All":
            filtered_df = filtered_df[filtered_df['currentPatrolState'] == self.state_var.get()]

        # if isNight != "All":
        #     if isNight == "day":
        #         filtered_df = filtered_df[filtered_df['isNight'] == 0]
        #     else:
        #         filtered_df = filtered_df[filtered_df['isNight'] == 1]
        

        # Wybór pomiędzy jednostkami czasu
        if self.time_unit_var.get() == "Hour":
            grouped_df = filtered_df.groupby(['Hour']).size().reset_index(name='patrolCount')
        elif self.time_unit_var.get() == "Minute":
            grouped_df = filtered_df.groupby(['Hour', 'Minute']).size().reset_index(name='patrolCount')
        elif self.time_unit_var.get() == "Second":
            grouped_df = filtered_df.groupby(['Hour', 'Minute', 'simulationTime[s]']).size().reset_index(name='patrolCount')                

        if self.presentation_mode == 'chart':
            self.draw_chart(grouped_df)
        elif self.presentation_mode == 'table':
            self.create_table(filtered_df)
            
        self.analyse(grouped_df)

        # Utworzenie buttona do exportu danych
        button = ttk.Button(self.export_frame, text="Export data", width=16, command=lambda: export_to_csv(filtered_df, f"Context sharing - {self.time_unit_var.get()} - {self.city_var.get()}"))
        button.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

    # Rysowanie wykresu
    def draw_chart(self, grouped_data):
        # Wyczyszczenie frame
        for widget in self.frame_chart.winfo_children():
            widget.destroy()
        
        # Ustawienia wykresu
        fig, ax = plt.subplots(figsize=(16, 9))

        # Ustawienie koloru tła
        fig.patch.set_facecolor('#313131')
        fig.patch.set_alpha(1.0)
        ax.patch.set_facecolor('#313131')
        ax.patch.set_alpha(0.2)

        # Zmiana koloru czcionek na biały
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color('white')

        # Zmiana koloru etykiet osi
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

        # Zmiana koloru tytułu
        ax.title.set_color('white')

        # Zmiana koloru podziałek
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        # Zmiana koloru linii
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')

        if self.time_unit_var.get() == "Hour":
            # Wykres dla godzin
            plt.stem(grouped_data['Hour'], grouped_data['patrolCount'])
            plt.title(f'Suma ilości patroli w zależności od godziny ({self.start_time} - {self.end_time}) for {self.city_var.get()}')
            plt.xlabel('Godzina')
            plt.ylabel('Ilość patroli')
            ax = plt.gca()
            ax.xaxis.set_major_locator(MultipleLocator(1))
        elif self.time_unit_var.get() == "Minute":
            # Wykres dla minut
            plt.stem(grouped_data['Hour'] * 60 + grouped_data['Minute'], grouped_data['patrolCount'])
            plt.title(f'Suma ilości patroli w zależności od minuty ({self.start_time} - {self.end_time}) for {self.city_var.get()}')
            plt.xlabel('Czas symulacji [minuty]')
            plt.ylabel('Ilość patroli')
        elif self.time_unit_var.get() == "Second":
            # Wykres dla sekund
            plt.stem(grouped_data['Hour'] * 3600 + grouped_data['Minute'] * 60 + grouped_data['simulationTime[s]'], grouped_data['patrolCount'])
            plt.title(f'Suma ilości patroli w zależności od sekundy ({self.start_time} - {self.end_time}) for {self.city_var.get()}')
            plt.xlabel('Czas symulacji [sekundy]')
            plt.ylabel('Ilość patroli')

        container_frame = ttk.Frame(self.frame_chart)
        container_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Osadzenie wykresu w interfejsie tkinter i narysowanie
        canvas = FigureCanvasTkAgg(fig, master=container_frame)
        canvas.draw()

        # Utworzenie buttona do exportu wykresu
        button = ttk.Button(self.export_frame, text="Export chart", width=16, command=lambda: export_plot_to_image(fig, f"Context sharing - {self.time_unit_var.get()} - {self.city_var.get()}"))
        button.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        # Zamyknięcie wykresu po użyciu
        plt.close(fig)

        # Dodanie paseka narzędziowego
        toolbar = NavigationToolbar2Tk(canvas, container_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Umieszczenie canvas w grid w frame
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


    # Analiza danych
    def analyse(self, data):
        # Wyczyszczenie frame
        for widget in self.frame_analise.winfo_children():
            widget.destroy()

        # Utworzenie Treeview
        tree = ttk.Treeview(self.frame_analise, columns=list(data.columns), show="headings")

        # Dodanie nagłówków do kolumn
        for column in data.columns:
            tree.heading(column, text=column)
            tree.column(column, anchor="center")

        # Dodanie danych do Treeview
        for index, row in data.iterrows():
            tree.insert("", tk.END, values=list(row))

        tree.pack(side="left", fill="both", expand=True)

        ###########################
        ########## WYKRES #########
        ###########################
        # Rysowanie histogramu
        fig, ax = plt.subplots(figsize=(5, 3))

        # Ustawienie koloru tła
        fig.patch.set_facecolor('#313131')
        fig.patch.set_alpha(1.0)
        ax.patch.set_facecolor('#313131')
        ax.patch.set_alpha(0.2)

        # Zmiana koloru czcionek na biały
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color('white')

        # Zmiana koloru etykiet osi
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

        # Zmiana koloru tytułu
        ax.title.set_color('white')

        # Zmiana koloru podziałek
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        # Zmiana koloru linii
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
    
        if self.time_unit_var.get() == "Hour":
            # Wykres dla godzin
            ax.plot(data['Hour'], data['patrolCount'])
            ax.set_title(f'Suma ilości patroli w zależności od godziny')
            ax.set_xlabel('Godzina')
            ax.set_ylabel('Ilość patroli')
            ax = plt.gca()
            ax.xaxis.set_major_locator(MultipleLocator(1))
        elif self.time_unit_var.get() == "Minute":
            # Wykres dla minut
            ax.plot(data['Hour'] * 60 + data['Minute'], data['patrolCount'])
            ax.set_title(f'Suma ilości patroli w zależności od minuty')
            ax.set_xlabel('Czas symulacji [minuty]')
            ax.set_ylabel('Ilość patroli')
        elif self.time_unit_var.get() == "Second":
            # Wykres dla sekund
            ax.plot(data['Hour'] * 3600 + data['Minute'] * 60 + data['simulationTime[s]'], data['patrolCount'])
            ax.set_title(f'Suma ilości patroli w zależności od sekundy')
            ax.set_xlabel('Czas symulacji [sekundy]')
            ax.set_ylabel('Ilość patroli')

        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.25, top=0.85)

        # Osadzenie wykresu w interfejsie tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.frame_analise)
        canvas.draw()

        # Zamyknięcie wykresu po użyciu
        plt.close(fig)

        canvas.get_tk_widget().pack(side=tk.LEFT, padx=10, pady=10)


    # Widok tabeli
    def create_table(self, data):
        # Wyczyszczenie frame
        for widget in self.frame_chart.winfo_children():
            widget.destroy()

        # Utworzenie Treeview
        tree = ttk.Treeview(self.frame_chart, columns=list(data.columns), show="headings")

        # Dodanie nagłówków do kolumn
        for column in data.columns:
            tree.heading(column, text=column)
            tree.column(column, anchor="center")

        # Dodanie danych do Treeview
        for index, row in data.iterrows():
            tree.insert("", tk.END, values=list(row))

        # Dodanie paska przewijania
        scrollbar_y = ttk.Scrollbar(self.frame_chart, orient="vertical", command=tree.yview)
        scrollbar_x = ttk.Scrollbar(self.frame_chart, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        # Przypisanie polecenia przewijania do pasków
        scrollbar_y.config(command=tree.yview)
        scrollbar_x.config(command=tree.xview)

        scrollbar_y.pack(side="right", fill="y")
        scrollbar_x.pack(side="bottom", fill="x")

        # Pakowanie Treeview i paska przewijania
        tree.pack(side="left", fill="both", expand=True)