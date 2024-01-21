import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
from src.dataOperations.export_chart import export_plot_to_image

from src.dataOperations.export_data import export_to_csv
from src.dataOperations.load_data import load_data


class ChangingStateDetailsVisualizer():
    def __init__(self, frame1, frame2, selected_cities):
        super().__init__()

        self.frame1 = frame1
        self.frame2 = frame2
        # self.selected_cities = selected_cities
        # self.data = data

        chart_topic = "Changing State Details"

        self.data = load_data(selected_cities, chart_topic)

        self.selected_cities = list(selected_cities)

        ###########################
        ######### ZMIENNE #########
        ###########################
        self.city_var = tk.StringVar()
        self.state_var = tk.StringVar()
        self.patrol_var = tk.StringVar()
        self.state_dropdown = tk.StringVar()
        self.average_var = tk.BooleanVar()
        self.patrol_dropdown = tk.StringVar()
        self.direction = tk.StringVar()
        self.direction = 'previousPatrolState'

        self.state_radios = []
        self.city_radios = []
        self.city_state_radios = []
        self.presentation_mode = 'chart'

        self.states = [
            "PATROLLING",
            "FIRING",
            "INTERVENTION",
            "NEUTRALIZED",
            "CALCULATING_PATH",
            "RETURNING_TO_HQ",
            "TRANSFER_TO_INTERVENTION",
            "TRANSFER_TO_FIRING"
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

        mode_radio = ttk.Radiobutton(set_city_frame, text="All", value="All", variable=self.city_var, command=lambda: self.after_city_set(set_patrol_frame))
        mode_radio.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        for i, city in enumerate(self.selected_cities):
            mode_radio = ttk.Radiobutton(set_city_frame, text=city, value=city, variable=self.city_var, command=lambda: self.after_city_set(set_patrol_frame))
            mode_radio.grid(row=i+1, column=0, padx=5, pady=5, sticky="nsew")
        self.city_var.set("All")
        
        set_state_frame = ttk.LabelFrame(self.options_frame, text="Set state")
        set_state_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        mode_choose_direction = ttk.Checkbutton(set_state_frame, text="Set previous | Set next", style="Switch", command=lambda: self.toggle_choose_direction(mode_choose_direction))
        mode_choose_direction.grid(row=0, column=0, padx=5, pady=10, sticky="nsew")

        set_patrol_frame = ttk.LabelFrame(self.options_frame, text="Set type")
        set_patrol_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")

        state_dropdown = ttk.Combobox(set_state_frame, values=self.states, width=28, textvariable=self.state_dropdown)
        state_dropdown.current(0)
        state_dropdown.bind("<<ComboboxSelected>>", self.prepare_data)
        state_dropdown.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        average_state_frame = ttk.LabelFrame(self.options_frame, text="Draw avarage line")
        average_state_frame.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")

        average_checkbox = ttk.Checkbutton(average_state_frame, text="Show Average on Chart", variable=self.average_var, command=self.prepare_data)
        average_checkbox.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        display_frame = ttk.LabelFrame(self.options_frame, text="Display")
        display_frame.grid(row=4, column=0, padx=20, pady=10, sticky="nsew")

        mode_switch_presentation = ttk.Checkbutton(display_frame, text="Chart | Table", style="Switch", command=lambda: self.toggle_mode_presentation(mode_switch_presentation))
        mode_switch_presentation.grid(row=0, column=0, padx=5, pady=10, sticky="nsew")

        self.after_city_set(set_patrol_frame)
    

    # Funkcja po wybraniu miasta
    def after_city_set(self, set_patrol_frame):
        if self.city_var.get() == 'All':
            all_patrols_id = self.data['patrolID'].unique()
        else:
            data = self.data[self.data['City'] == self.city_var.get()]
            all_patrols_id = data['patrolID'].unique()
        
        patrols_numbers = list(all_patrols_id)

        patrol_switch = ttk.Checkbutton(set_patrol_frame, text="All Patrol | One Patrol", style="Switch", command=lambda: self.toggle_patrol(patrols_numbers, set_patrol_frame, patrol_switch))
        patrol_switch.grid(row=0, column=0, padx=5, pady=10, sticky="nsew")
        
        children = set_patrol_frame.winfo_children()
        for child in children:
            info = child.grid_info()
            if info['row'] == 1:
                child.destroy()
        
        self.toggle_patrol(patrols_numbers, set_patrol_frame, patrol_switch)

        
    # Obsługa Switch'a
    def toggle_patrol(self, all_patrols_id, frame, mode_switch):
        self.patrol_var = ""
        children = frame.winfo_children()
        for child in children:
            info = child.grid_info()
            if info['row'] == 1:
                child.destroy()

        if mode_switch.instate(["selected"]):
            self.patrol_var = "Selected"
            patrol_dropdown = ttk.Combobox(frame, values=all_patrols_id, width=28, textvariable=self.patrol_dropdown)
            patrol_dropdown.current(0)
            patrol_dropdown.bind("<<ComboboxSelected>>", self.prepare_data)
            patrol_dropdown.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        else:
            self.patrol_var = "All"

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


    # Obsługa Switch'a
    def toggle_choose_direction(self, mode_switch):
        for widget in self.frame_chart.winfo_children():
            widget.destroy()

        if mode_switch.instate(["selected"]):
            self.direction = 'currentPatrolState'
        else:
            self.direction = 'previousPatrolState'

        self.prepare_data()


    # Przygotowanie danych
    def prepare_data(self, event=None):
        if self.city_var.get() != '' and self.state_dropdown.get() != '' and self.direction != '': 
            if self.city_var.get() == 'All':
                if self.patrol_var == 'All':
                    grouped_data = self.data[self.data[self.direction] == self.state_dropdown.get()]
                elif self.patrol_var == 'Selected':
                    grouped_data = self.data[(self.data['patrolID'] == self.patrol_dropdown.get()) & (self.data[self.direction] == self.state_dropdown.get())]
            else:
                if self.patrol_var == 'All':
                    grouped_data = self.data[(self.data[self.direction] == self.state_dropdown.get()) & (self.data['City'] == self.city_var.get())]
                elif self.patrol_var == 'Selected':
                    grouped_data = self.data[(self.data['patrolID'] == self.patrol_dropdown.get()) & (self.data[self.direction] == self.state_dropdown.get()) & (self.data['City'] == self.city_var.get())]                     

        if self.presentation_mode == 'chart':
            self.draw_chart(grouped_data)
        elif self.presentation_mode == 'table':
            self.create_table(grouped_data)
            
        self.analyse(grouped_data)

        # Utworzenie buttona do exportu danych
        button = ttk.Button(self.export_frame, text="Export data", width=16, command=lambda: export_to_csv(grouped_data, f"Changing State Details - {self.city_var.get()} - {self.state_var.get()} - {self.direction}"))
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

        if self.direction == 'currentPatrolState':
            state = 'previousPatrolState'
        else:
            state = 'currentPatrolState'


        # Sporządzenie histogramu dla poprzednich stanów
        sns.countplot(y=state, data=grouped_data, hue=state, palette='viridis', orient='h', legend=False)

        plt.subplots_adjust(left=0.2, right=0.95)

        # Obliczenie średniej liczby przejść do innych stanów
        mean_previous_state_counts = grouped_data[state].value_counts().mean()

        # Wyświetlenie średniej liczby przejść do innych stanów
        if self.average_var.get():
            plt.axvline(mean_previous_state_counts, color='red', linestyle='dashed', linewidth=2,
                        label='Average Transition Count')
            plt.legend()

        if self.city_var.get() != 'All':
            city = (f'for {self.city_var.get()}')
        else:
            city = ""

        if  self.patrol_var == "Selected" and self.city_var.get() != 'All':
            patrolId = (f'and patrol id: {self.patrol_dropdown.get()}')
        elif self.patrol_var == "Selected" and self.city_var.get() == 'All':
            patrolId = (f'for patrol id: {self.patrol_dropdown.get()}')
        else:
            patrolId = ""

        # Dodaj etykiety dla osi x i y oraz tytuł wykresu
        if self.direction == 'currentPatrolState':
            ax.set_title(f'States before {self.state_dropdown.get()} {city} {patrolId}')
        else:
            ax.set_title(f'States after {self.state_dropdown.get()} {city} {patrolId}')
        ax.set_xlabel('Number of Occurrences')
        ax.set_ylabel('Previous State')

        container_frame = ttk.Frame(self.frame_chart)
        container_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Osadzenie wykresu w interfejsie tkinter i narysowanie
        canvas = FigureCanvasTkAgg(fig, master=container_frame)
        canvas.draw()

        # Utworzenie buttona do exportu wykresu
        button = ttk.Button(self.export_frame, text="Export chart", width=16, command=lambda: export_plot_to_image(fig, f"Changing State Details - {self.city_var.get()} - {self.state_var.get()} - {self.direction}"))
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

        groupby = self.direction
        if self.direction == 'currentPatrolState':
            count_by = 'previousPatrolState'
        else:
            count_by = 'currentPatrolState'

        # Usunięcie danych, gdzie simulationTime[s] wynosi 0
        data = data[data["simulationTime[s]"] != 0]

        # Analiza danych
        grouped_data = data.groupby([groupby, count_by]).size().reset_index(name="count")

        all = grouped_data['count'].sum()

        # Dodanie kolumny z procentowym udziałem
        grouped_data['percentage'] = grouped_data.groupby(count_by)['count'].transform(lambda x: x / all * 100)

        # Utwórzenie drzewa do wyświetlania danych
        tree = ttk.Treeview(self.frame_analise, columns=["group", "previousPatrolState", "count", "percentage"], show="headings")

        # Dodanie kolumny
        tree.heading("group", text="State")
        tree.heading("previousPatrolState", text="Next state")
        tree.heading("count", text=f"Count by {count_by}")
        tree.heading("percentage", text="Percentage")

        tree.column("#1", anchor="center")
        tree.column("#2", anchor="center")
        tree.column("#3", anchor="center")
        tree.column("#4", anchor="center")

        for index, row in grouped_data.iterrows():
            tree.insert("", "end", values=(row['previousPatrolState'], row['currentPatrolState'], row["count"],  f"{row['percentage']:.2f}%"))

        tree.pack(side="left", fill="both", expand=True)

        ###########################
        ########## WYKRES #########
        ###########################
        # Tworzenie figury i osi
        fig, ax_pie = plt.subplots(figsize=(5, 4), subplot_kw=dict(aspect="equal"))
        fig.patch.set_facecolor('#313131')
        fig.patch.set_alpha(1.0)

        # Rysowanie wykresu kołowego
        wedges, texts, autotexts = ax_pie.pie(grouped_data['count'], labels=grouped_data[count_by].unique(), autopct='%1.1f%%', startangle=140)

        # Dodanie legendy
        legend = ax_pie.legend(wedges, grouped_data[count_by].unique(), title="Legend:", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
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
        ax_pie.set_title(f'{count_by} to {groupby}')
        ax_pie.title.set_color('#217346')
        ax_pie.title.set_size(9)

        # Zmiana koloru tekstu na biały
        for text_obj in ax_pie.texts + autotexts:
            text_obj.set_color('white')

        plt.setp(autotexts, weight="bold", size=8)

        fig.subplots_adjust(left=0.0, right=0.55, bottom=0.0, top=0.85)

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