import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from src.dataOperations.export_chart import export_plot_to_image
from src.dataOperations.export_data import export_to_csv
from src.dataOperations.load_data import load_data

types_of_patrol = {
    "Names": [
        "All",
        "Patrols",
        "Patrolling Patrols",
        "Calculating Path Patrols",
        "Transfer To Intervention Patrols",
        "Transfer To Firing Patrols",
        "Intervention Patrols",
        "Firing Patrols",
        "Incidents",
        "Night"
    ],
    "names_in_df": [
        'all',
        'amountOfPatrols',
        'amountOfPatrollingPatrols',
        'amountOfCalculatingPathPatrols',
        'amountOfTransferToInterventionPatrols',
        'amountOfTransferToFiringPatrols',
        'amountOfInterventionPatrols',
        'amountOfFiringPatrols',
        'amountOfIncidents',
        'isNight'
    ]
}
# Lista kategorii
categories = ['Safe', 'Rather Safe', 'Not Safe']

class DistinctDetailsVisualizer:
    def __init__(self, frame1, frame2, selected_cities):
        super().__init__()

        self.frame1 = frame1
        self.frame2 = frame2
        # self.selected_cities = selected_cities
        # self.data = data

        chart_topic = "Distinct Details"

        self.data = load_data(selected_cities, chart_topic)

        self.selected_cities = list(selected_cities)

        ###########################
        ######### ZMIENNE #########
        ###########################
        self.city_var = tk.StringVar()
        self.state_var = tk.StringVar()

        self.city_state_radios = []
        self.canvas_dict = {}
        self.presentation_mode = 'chart'

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
        self.frame2.grid_columnconfigure(0, weight=2)
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

        city__frame = ttk.LabelFrame(self.options_frame, text="City")
        city__frame.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")

        mode_radio = ttk.Radiobutton(city__frame, text="All", value="All", variable=self.city_var, command=self.prepare_data)
        mode_radio.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        for i, city in enumerate(self.selected_cities):
            mode_radio = ttk.Radiobutton(city__frame, text=city, value=city, variable=self.city_var, command=self.prepare_data)
            mode_radio.grid(row=i+1, column=0, padx=5, pady=5, sticky="nsew")
        self.city_var.set("All")
        
        type__frame = ttk.LabelFrame(self.options_frame, text="Type")
        type__frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        type_dropdown = ttk.Combobox(type__frame, values=types_of_patrol['Names'], width=25, textvariable=self.state_var)
        type_dropdown.current(0)
        type_dropdown.bind("<<ComboboxSelected>>", self.prepare_data)
        type_dropdown.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        display_frame = ttk.LabelFrame(self.options_frame, text="Display")
        display_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")

        mode_switch_presentation = ttk.Checkbutton(display_frame, text="Chart | Table", style="Switch", command=lambda: self.toggle_mode_presentation(mode_switch_presentation))
        mode_switch_presentation.grid(row=0, column=0, padx=5, pady=10, sticky="nsew")

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
        if self.city_var.get() != '':
            if self.city_var.get() == 'All':
                selected_data = self.data 
            else:
                selected_data = self.data[self.data['City'] ==  self.city_var.get()]

            index_of_selected_type = types_of_patrol["Names"].index( self.state_var.get())
            names_in_df = types_of_patrol["names_in_df"][index_of_selected_type]

            if self.state_var.get() == 'All':
                grouped_data = selected_data.groupby('districtSafetyLevel').agg({
                    'amountOfPatrols': 'sum',
                    'amountOfPatrollingPatrols': 'sum',
                    'amountOfCalculatingPathPatrols': 'sum',
                    'amountOfTransferToInterventionPatrols': 'sum',
                    'amountOfTransferToFiringPatrols': 'sum',
                    'amountOfInterventionPatrols': 'sum',
                    'amountOfFiringPatrols': 'sum',
                    # 'amountOfReturningToHqPatrols': 'sum',
                    'amountOfIncidents': 'sum',
                    'isNight' : 'sum'
                })
            else:
                grouped_data = selected_data.groupby('districtSafetyLevel').agg({
                    names_in_df: 'sum'
                })

        if self.presentation_mode == 'chart':
            self.draw_chart(grouped_data, names_in_df)
        elif self.presentation_mode == 'table':
            self.create_table(selected_data)

        self.analyse(grouped_data, names_in_df, "districtSafetyLevel")

        # Utworzenie buttona do exportu danych
        button = ttk.Button(self.export_frame, text="Export data", command=lambda: export_to_csv(selected_data, f"Distinct Details - {self.city_var.get()} - {self.state_var.get()}"))
        button.grid(row=0, column=0, padx=15, pady=5, sticky="nsew")


    # Rysowanie wykresu
    def draw_chart(self, data, type_of_patrol):
    # Wyczyszczenie frame
        for widget in self.frame_chart.winfo_children():
            widget.destroy()

        if type_of_patrol == 'all':
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 9))

            fig.patch.set_facecolor('#313131')
            fig.patch.set_alpha(1.0)

            for i, column in enumerate(data.columns):
                row, col = divmod(i, 3)

                if i < len(data.columns):
                    ax_pie = axes[row, col]
                    wedges, texts, autotexts = ax_pie.pie(data[column], labels=categories, autopct='%1.1f%%', startangle=140)

                    if i == 5:
                        legend = ax_pie.legend(wedges, categories, title="Legend:", loc="center left", bbox_to_anchor=(1.5, 0, 0.5, 1))
                        legend.get_title().set_color('white')
                        legend.set_frame_on(True)
                        legend.get_frame().set_facecolor('#595959')
                        legend.get_frame().set_alpha(1.0)

                        for text_obj in legend.get_texts():
                            text_obj.set_color('white')

                    # Ukrycie etykiet
                    for text in texts:
                        text.set_visible(False)

                    # Dodanie tytułu
                    ax_pie.set_title(types_of_patrol['Names'][i + 1])
                    ax_pie.title.set_color('#217346')

                    # Zmiana koloru tekstu na biały
                    for text_obj in ax_pie.texts + autotexts:
                        text_obj.set_color('white')

                plt.setp(autotexts, weight="bold", size=8)

        else:
            # Tworzenie figury i osi
            fig, ax_pie = plt.subplots(figsize=(16, 9), subplot_kw=dict(aspect="equal"))
            fig.patch.set_facecolor('#313131')
            fig.patch.set_alpha(1.0)

            # Rysowanie wykresu kołowego
            wedges, texts, autotexts = ax_pie.pie(data[type_of_patrol], labels=categories, autopct='%1.1f%%', startangle=140)

            # Dodanie legendy
            legend = ax_pie.legend(wedges, categories, title="Legend:", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            legend.get_title().set_color('white')
            legend.get_title().set_size(13)
            for text_obj in legend.get_texts():
                text_obj.set_color('white')
                text_obj.set_size(13)

            # Ukrycie etykiet
            for text in texts:
                text.set_visible(False)

            # Zmiana koloru tła legendy
            legend.set_frame_on(True)
            legend.get_frame().set_facecolor('#595959')
            legend.get_frame().set_alpha(1.0)

            # Dodanie tytułu
            ax_pie.set_title(types_of_patrol["Names"][types_of_patrol["names_in_df"].index(type_of_patrol)])
            ax_pie.title.set_color('#217346')
            ax_pie.title.set_size(14)


            # Zmiana koloru tekstu na biały
            for text_obj in ax_pie.texts + autotexts:
                text_obj.set_color('white')

            plt.setp(autotexts, weight="bold", size=13)

        # Dostoswoanie pozycji subplotów
        fig.subplots_adjust(top=0.85, bottom=0, left=0.05, right=0.85, hspace=0.5, wspace=0.3)

        # Dodanie tytułu nad wszystkimi wykresami
        plt.suptitle(f'City: {self.city_var.get()}', fontsize=16, color='white')

        # Osadzenie wykresu w interfejsie tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.frame_chart)
        canvas.draw()  # Narysuj wykres na canvas

        # Utworzenie buttona do exportu wykresu
        button = ttk.Button(self.export_frame, text="Export chart", command=lambda: export_plot_to_image(fig, f"Distinct Details - {self.city_var.get()} - {self.state_var.get()}"))
        button.grid(row=1, column=0, padx=15, pady=5, sticky="nsew")

        # Zamyknięcie wykresu po użyciu
        plt.close(fig)

        # Dodanie paseka narzędziowego
        toolbar = NavigationToolbar2Tk(canvas, self.frame_chart)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Umieszczenie canvas w grid w frame
        canvas.get_tk_widget().pack(side=tk.LEFT, padx=10, pady=10)


    # Analiza danych
    def analyse(self, data, type_of_patrol, groupby):
        # Wyczyszczenie frame
        for widget in self.frame_analise.winfo_children():
            widget.destroy()

        # Dodanie sumy dla districtSafetyLevel
        data[type_of_patrol] = data.sum(axis=1)
        
        # Zresetuj indeksy
        grouped_data = data.reset_index()

        # Utworzenie drzewa do wyświetlania danych
        tree = ttk.Treeview(self.frame_analise, columns=["group", "sum"])
        tree["show"] = "headings"

        # Dodanie kolumn
        tree.heading("group", text=groupby)
        tree.heading("sum", text="Amount")

        tree.column("#1", anchor="w")
        tree.column("#2", anchor="center")

        for index, row in grouped_data.iterrows():
            tree.insert("", "end", values=(row[groupby], row[type_of_patrol]))

        tree.pack(side="left", fill="both", expand=True)

        ###########################
        ########## WYKRES #########
        ###########################
        # Rysowanie histogramu
        fig, ax = plt.subplots(figsize=(5, 5))

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

        ax.bar(grouped_data[groupby], grouped_data[type_of_patrol])
        ax.set_xlabel(groupby)
        ax.set_ylabel("Amount")
        ax.set_title("District Patrols Analysis")

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