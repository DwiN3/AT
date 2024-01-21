import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from src.dataOperations.export_chart import export_plot_to_image
from src.dataOperations.load_data import load_data

from src.dataOperations.export_data import export_to_csv


class PatrolDataVisualizer():
    def __init__(self, frame1, frame2, selected_cities):
        super().__init__()

        self.frame1 = frame1
        self.frame2 = frame2
        # self.selected_cities_ = selected_cities
        # self.data = data

        chart_topic = "First Patrol Data"

        self.data = load_data(selected_cities, chart_topic)

        self.selected_cities = list(selected_cities)

        ###########################
        ######### ZMIENNE #########
        ###########################
        self.city_var = tk.StringVar()
        self.state_var = tk.StringVar()

        self.city_state_radios = []
        self.presentation_mode = 'chart'

        self.patrol_state_list = [
            "CALCULATING_PATH",
            "PATROLLING",
            "FIRING",
            "INTERVENTION",
            "TRANSFER_TO_INTERVENTION"
        ]

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

        city_or_state_frame = ttk.LabelFrame(self.options_frame, text="City Or State")
        city_or_state_frame.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")

        mode_switch = ttk.Checkbutton(city_or_state_frame, text="City | State", style="Switch", command=lambda: self.toggle_mode(mode_switch, city_or_state_frame))
        mode_switch.grid(row=0, column=0, padx=5, pady=10, sticky="nsew")
        self.toggle_mode(mode_switch, city_or_state_frame)

        display_frame = ttk.LabelFrame(self.options_frame, text="Display")
        display_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        mode_switch_presentation = ttk.Checkbutton(display_frame, text="Chart | Table", style="Switch", command=lambda: self.toggle_mode_presentation(mode_switch_presentation))
        mode_switch_presentation.grid(row=0, column=0, padx=5, pady=10, sticky="nsew")

    
    # Obsługa Switch'a
    def toggle_mode(self, mode_switch, frame):
        self.city_var.set("")
        self.state_var.set("")

        for radio in self.city_state_radios:
            radio.destroy()

        if mode_switch.instate(["selected"]):
            for i, state in enumerate(self.patrol_state_list):
                mode_radio = ttk.Radiobutton(frame, text=state, value=state, variable=self.state_var, command=self.prepare_data)
                mode_radio.grid(row=i+1, column=0, padx=5, pady=5, sticky="nsew")
                self.city_state_radios.append(mode_radio)
            self.state_var.set(self.patrol_state_list[0])
        else:
            for i, city in enumerate(self.selected_cities):
                mode_radio = ttk.Radiobutton(frame, text=city, value=city, variable=self.city_var, command=self.prepare_data)
                mode_radio.grid(row=i+1, column=0, padx=5, pady=5, sticky="nsew")
                self.city_state_radios.append(mode_radio)
            self.city_var.set(self.selected_cities[0])
        
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
    def prepare_data(self):
        if self.city_var.get() != '':
            selected_data = self.data[self.data['City'] == self.city_var.get()]

            grouped_data = selected_data.groupby('patrolState')
            groupby = "patrolState"

        elif self.state_var.get() != '':
            selected_data = self.data[self.data['patrolState'] == self.state_var.get()]

            grouped_data = selected_data.groupby('City')
            groupby = "City"

        data_to_analyse = selected_data[selected_data["timeInState[s]"] != 0]

        if self.presentation_mode == 'chart':
            self.draw_chart(grouped_data)
        elif self.presentation_mode == 'table':
            self.create_table(selected_data)
            
        self.analyse(data_to_analyse, groupby)

        # Utworzenie buttona do exportu danych
        button = ttk.Button(self.export_frame, text="Export data", command=lambda: export_to_csv(data_to_analyse, f"First Patrol Data - {self.city_var.get()} - {self.state_var.get()}"))
        button.grid(row=0, column=0, padx=15, pady=5, sticky="nsew")


    # Rysowanie wykresu
    def draw_chart(self, grouped_data):
        # Wyczyszczenie frame
        for widget in self.frame_chart.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots()

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

        # Sporządzenie wykresów
        for state, group in grouped_data:
            plt.plot(group['simulationTime[s]'] / 3600, group['timeInState[s]'] / 60, label=state, linestyle='-')

        # Ustawienie przyrostów na osiach
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(5))
        
        # Dodanie legendy
        plt.legend()

        # Dodanie etykiet oraz tytułu wykresu
        plt.xlabel('Simulation Time [h]')
        plt.ylabel('Time in State [min]')

        if self.city_var.get() != '':
            title = self.city_var.get()
        elif self.state_var.get() != '':
            title = self.state_var.get()
        plt.title(f'Patrol State Simulation for {title}')

        container_frame = ttk.Frame(self.frame_chart)
        container_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Osadzenie wykresu w interfejsie tkinter i narysowanie
        canvas = FigureCanvasTkAgg(fig, master=container_frame)
        canvas.draw()

        # Utworzenie buttona do exportu wykresu
        button = ttk.Button(self.export_frame, text="Export chart", command=lambda: export_plot_to_image(fig, f"First Patrol Data - {self.city_var.get()} - {self.state_var.get()}"))
        button.grid(row=1, column=0, padx=15, pady=5, sticky="nsew")

        # Zamyknięcie wykresu po użyciu
        plt.close(fig)

        # Dodanie paseka narzędziowego
        toolbar = NavigationToolbar2Tk(canvas, container_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Umieszczenie canvas w grid w frame
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


    # Analiza danych
    def analyse(self, data, groupby):
        # Wyczyszczenie frame
        for widget in self.frame_analise.winfo_children():
            widget.destroy()

        # Utworzenie pola tekstowego
        tree = ttk.Treeview(self.frame_analise, columns=["group", "sum", "mean", "min", "max"])
        tree["show"] = "headings"

        # Dodanie kolumn
        tree.heading("group", text=groupby)
        tree.heading("sum", text="Sum time")
        tree.heading("mean", text="Mean time")
        tree.heading("min", text="Min time")
        tree.heading("max", text="Max time")

        tree.column("#1", anchor="w")
        tree.column("#2", anchor="center")
        tree.column("#3", anchor="center")
        tree.column("#4", anchor="center")
        tree.column("#5", anchor="center")

        for name, group in data.groupby(groupby)["timeInState[s]"]:
            total_seconds = group.sum()
            
            minutes = total_seconds // 60
            seconds = total_seconds % 60

            mean_seconds = group.mean() % 60
            min_seconds = group.min() % 60
            max_seconds = group.max() % 60

            sum_val = f"{int(minutes)} min {int(seconds)} s"
            mean_val = f"{int(group.mean() // 60)} min {int(mean_seconds)} s"
            min_val = f"{int(group.min() // 60)} min {int(min_seconds)} s"
            max_val = f"{int(group.max() // 60)} min {int(max_seconds)} s"

            tree.insert("", "end", values=(name, sum_val, mean_val, min_val, max_val), tags=(name,))

        tree.pack(side="left", fill="both", expand=True)


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
        scrollbar = ttk.Scrollbar(self.frame_chart, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        # Pakowanie Treeview i paska przewijania
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")