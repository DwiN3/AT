import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from src.dataOperations.export_chart import export_plot_to_image
import pandas as pd
import seaborn as sns
from src.dataOperations.export_data import export_to_csv
from src.dataOperations.load_data import load_data


class PatrolDistanceVisualizer():
    def __init__(self, frame1, frame2, selected_cities):
        super().__init__()

        self.frame1 = frame1
        self.frame2 = frame2

        chart_topic = "Patrol Distance To Reach Incident"

        self.data = load_data(selected_cities, chart_topic)

        self.selected_cities = list(selected_cities)

        ###########################
        ######### ZMIENNE #########
        ###########################
        self.city_var = tk.StringVar()
        self.mode_dropdown_var = tk.StringVar()
        self.distinct_var = tk.StringVar()
        self.firing_id_var = tk.StringVar()
        self.safety_level_var = tk.StringVar()
        self.displaySafetyLevel = tk.BooleanVar()

        self.city_state_radios = []
        self.presentation_mode = 'chart'

        self.safety_levels = [
            "Safe",
            "NotSafe",
            "RatherSafe"
        ]

        self.modes = [
            'Sum',
            'Mean'
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

        set_city_frame = ttk.LabelFrame(self.options_frame, text="Set city")
        set_city_frame.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")

        mode_radio = ttk.Radiobutton(set_city_frame, text="All", value="All", variable=self.city_var, command=self.prepare_data)
        mode_radio.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        for i, city in enumerate(self.selected_cities):
            mode_radio = ttk.Radiobutton(set_city_frame, text=city, value=city, variable=self.city_var, command=self.prepare_data)
            mode_radio.grid(row=i + 1, column=0, padx=5, pady=5, sticky="nsew")
        self.city_var.set("All")

        set_mode_frame = ttk.LabelFrame(self.options_frame, text="Set frame mode")
        set_mode_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        for i, mode in enumerate(self.modes):
            mode_radio = ttk.Radiobutton(set_mode_frame, text=mode, value=mode, variable=self.mode_dropdown_var, command=self.prepare_data)
            mode_radio.grid(row=i + 1, column=0, padx=5, pady=5, sticky="nsew")
        self.mode_dropdown_var.set("Sum")

        self.prepare_data()

        display_frame = ttk.LabelFrame(self.options_frame, text="Display")
        display_frame.grid(row=4, column=0, padx=20, pady=10, sticky="nsew")

        mode_switch_presentation = ttk.Checkbutton(display_frame, text="Chart | Table", style="Switch", command=lambda: self.toggle_mode_presentation(mode_switch_presentation))
        mode_switch_presentation.grid(row=0, column=0, padx=5, pady=10, sticky="nsew")


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
        df = self.data.copy()

        # Wybierz dane na podstawie miasta
        if self.city_var.get() != "All":
            df = df[df["City"] == self.city_var.get()]

        # Konwersja kolumny "distanceOfSummonedPatrol[m]" na numeryczne wartości
        df['distanceOfSummonedPatrol[m]'] = df['distanceOfSummonedPatrol[m]'].replace({',': '.'}, regex=True).astype(float)
        df['distanceOfSummonedPatrol[m]'] /= 1000

        # Grupujemy dane według kategorii "districtSafetyLevel" i obliczamy średnią lub sumę dla "distanceOfSummonedPatrol[m]"
        if self.mode_dropdown_var.get() == "Sum":
            grouped_df = df.groupby('districtSafetyLevel')['distanceOfSummonedPatrol[m]'].sum()
        else:
            grouped_df = df.groupby('districtSafetyLevel')['distanceOfSummonedPatrol[m]'].mean()

        # Wybór interesujących kolumn
        selected_columns = [
            "districtSafetyLevel",
            "distanceOfSummonedPatrol[m]"
        ]

        df_to_analise = df[selected_columns].groupby("districtSafetyLevel").sum()

        df_to_analise["sumOfDistances"] = df.groupby('districtSafetyLevel')['distanceOfSummonedPatrol[m]'].sum()
        df_to_analise["meanOfDistances"] = df.groupby('districtSafetyLevel')['distanceOfSummonedPatrol[m]'].mean()

        # Dodanie kolumny z udziałem procentowym dystansów
        df_to_analise["percentageOfTotalDistance"] = df_to_analise['sumOfDistances'] / df_to_analise['sumOfDistances'].sum() * 100

        df_to_analise = df_to_analise.reset_index()  # Reset index

        if self.presentation_mode == 'chart':
            self.draw_chart(grouped_df)
            self.analyse(df_to_analise)
        elif self.presentation_mode == 'table':
            self.create_table(grouped_df)
        
        # Utworzenie buttona do exportu danych
        button = ttk.Button(self.export_frame, text="Export data", command=lambda: export_to_csv(df, f"{self.mode_dropdown_var.get()} Patrol Distance To Reach Incident - {self.city_var.get()}"))
        button.grid(row=0, column=0, padx=15, pady=5, sticky="nsew")


    # Rysowanie wykresu
    def draw_chart(self, filtered_df):
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

        # Stworzenie wykresu za pomocą seaborn
        ax = sns.barplot(x=filtered_df.values, y=filtered_df.index, hue=filtered_df.index, palette='viridis')

        plt.subplots_adjust(left=0.2, right=0.95)

        # Dodanie etykiet do słupków
        for index, value in enumerate(filtered_df.values):
            plt.text(value, index, f'{value:.2f}', ha='left', va='center', color='white')

        plt.title(f"{self.mode_dropdown_var.get()} Patrols Distance in States for {self.city_var.get()}")
        plt.xlabel(f"{self.mode_dropdown_var.get()} Value")
        plt.ylabel("State Name")
        
        container_frame = ttk.Frame(self.frame_chart)
        container_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Osadzenie wykresu w interfejsie tkinter i narysowanie
        canvas = FigureCanvasTkAgg(fig, master=container_frame)
        canvas.draw()

        # Utworzenie buttona do exportu wykresu
        button = ttk.Button(self.export_frame, text="Export chart", command=lambda: export_plot_to_image(fig, f"{self.mode_dropdown_var.get()} Patrol Distance To Reach Incident - {self.city_var.get()}"))
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
    def analyse(self, data):        
        # Wyczyszczenie frame
        for widget in self.frame_analise.winfo_children():
            widget.destroy()

        # Utwórzenie drzewa do wyświetlania danych
        tree = ttk.Treeview(self.frame_analise, columns=["level", "totaldistance", "meandistance", "percentage"], show="headings")

        # Dodanie kolumny
        tree.heading("level", text="Level")
        tree.heading("totaldistance", text="Sum of Distances")
        tree.heading("meandistance", text="Mean of Distances")
        tree.heading("percentage", text="Percentage")

        tree.column("#1", anchor="center")
        tree.column("#2", anchor="center")
        tree.column("#3", anchor="center")
        tree.column("#4", anchor="center")

        for index, row in data.iterrows():
            tree.insert("", "end", values=(row['districtSafetyLevel'], f"{row['sumOfDistances']:.2f}km", f"{row['meanOfDistances']:.2f}km", f"{row['percentageOfTotalDistance']:.2f}%"))

        tree.pack(side="left", fill="both", expand=True)

        ###########################
        ########## WYKRES #########
        ###########################
        # Tworzenie figury i osi
        fig, ax_pie = plt.subplots(figsize=(5, 4), subplot_kw=dict(aspect="equal"))
        fig.patch.set_facecolor('#313131')
        fig.patch.set_alpha(1.0)

        # Rysowanie wykresu kołowego
        wedges, texts, autotexts = ax_pie.pie(data['sumOfDistances'], labels=data['districtSafetyLevel'].unique(), autopct='%1.1f%%', startangle=140)

        # Dodanie legendy
        legend = ax_pie.legend(wedges, data['districtSafetyLevel'].unique(), title="Legend:", loc="upper center", bbox_to_anchor=(0.5, -0.1))
        legend.get_title().set_color('white')
        legend.get_title().set_size(10)
        for text_obj in legend.get_texts():
            text_obj.set_color('white')
            text_obj.set_size(9)

        # Ukrycie etykiet
        for text in texts:
            text.set_visible(False)

        # Zmiana koloru tła legendy
        legend.set_frame_on(True)
        legend.get_frame().set_facecolor('#595959')
        legend.get_frame().set_alpha(1.0)

        # Dodanie tytułu
        ax_pie.set_title('Percentage of kilometers')
        ax_pie.title.set_color('#217346')
        ax_pie.title.set_size(9)

        # Zmiana koloru tekstu na biały
        for text_obj in ax_pie.texts + autotexts:
            text_obj.set_color('white')

        plt.setp(autotexts, weight="bold", size=8)

        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.3, top=0.90)

        # Osadzenie wykresu w interfejsie tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.frame_analise)
        canvas.draw()

        # Zamyknięcie wykresu po użyciu
        plt.close(fig)

        # Umieszczenie canvas w grid w frame
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