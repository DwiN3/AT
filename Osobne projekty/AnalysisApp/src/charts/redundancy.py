import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from src.dataOperations.export_chart import export_plot_to_image
import pandas as pd
import seaborn as sns
from src.dataOperations.export_data import export_to_csv
from src.dataOperations.load_data import load_data


class RedundancyVisualizer():
    def __init__(self, frame1, frame2, selected_cities):
        super().__init__()

        self.frame1 = frame1
        self.frame2 = frame2

        chart_topic = "Firings Details"

        self.data = load_data(selected_cities, chart_topic)

        self.selected_cities = list(selected_cities)

        ###########################
        ######### ZMIENNE #########
        ###########################
        self.city_var = tk.StringVar()
        self.mode_dropdown_var = tk.StringVar()
        self.distinct_var = tk.StringVar()
        self.firing_id_var = tk.StringVar()
        self.type_of_patrol_var = tk.StringVar()

        self.city_state_radios = []
        self.presentation_mode = 'chart'

        self.modes = [
            "Amount",
            "Mean",
            # "SafetyLevel"
        ]

        self.types_patrol = [
            "All",
            "Generally Required Patrols",
            "Solving Patrols",
            "Reaching Patrols",
            "Called Patrols"
        ]

        self.safety_levels = [
            "Safe",
            "NotSafe",
            "RatherSafe"
        ]
            
        self.columns_to_plot = [
            "generallyRequiredPatrols",
            "arrivedPatrols",
            "solvingPatrols",
            "reachingPatrols(including 'called')",
            "calledPatrols"
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

        mode_radio = ttk.Radiobutton(set_city_frame, text="All", value="All", variable=self.city_var, command=self.set_firing_id_or_district)
        mode_radio.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        for i, city in enumerate(self.selected_cities):
            mode_radio = ttk.Radiobutton(set_city_frame, text=city, value=city, variable=self.city_var, command=self.set_firing_id_or_district)
            mode_radio.grid(row=i+1, column=0, padx=5, pady=5, sticky="nsew")
        self.city_var.set("All")

        set_mode_frame = ttk.LabelFrame(self.options_frame, text="Set frame mode")
        set_mode_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        mode_dropdown = ttk.Combobox(set_mode_frame, values=self.modes, width=28, textvariable=self.mode_dropdown_var)
        mode_dropdown.current(0)
        mode_dropdown.bind("<<ComboboxSelected>>", self.on_combobox_selected)
        mode_dropdown.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.set_firing_id_or_district()

        display_frame = ttk.LabelFrame(self.options_frame, text="Display")
        display_frame.grid(row=4, column=0, padx=20, pady=10, sticky="nsew")

        mode_switch_presentation = ttk.Checkbutton(display_frame, text="Chart | Table", style="Switch", command=lambda: self.toggle_mode_presentation(mode_switch_presentation))
        mode_switch_presentation.grid(row=0, column=0, padx=5, pady=10, sticky="nsew")


    # Wybór odpowiedich funkjci po wybraniu combobox
    def on_combobox_selected(self, event):        
        children = self.options_frame.winfo_children()
        for child in children:
            info = child.grid_info()
            if info['row'] == 2:
                child.destroy()

        self.set_firing_id_or_district()
        self.prepare_data()


    # Wybór pomiędzy firing a distinct
    def set_firing_id_or_district(self):
        set_frame = ttk.LabelFrame(self.options_frame, text="Set firing | district")
        set_frame.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")

        mode_switch = ttk.Checkbutton(set_frame, text="District | Firing", style="Switch", command=lambda: self.toggle_mode(mode_switch, set_frame))
        mode_switch.grid(row=0, column=0, padx=5, pady=10, sticky="nsew")
        self.toggle_mode(mode_switch, set_frame)


    # Obsługa Switch'a
    def toggle_mode(self, mode_switch, frame):
        for radio in self.city_state_radios:
            radio.destroy()

        if mode_switch.instate(["selected"]):
            self.distinct_var.set('All')

            if self.city_var.get() == 'All':
                all_firing_id = self.data['firingID'].value_counts()[self.data['firingID'].value_counts() >= 2].index
            else:
                city_data = self.data[self.data['City'] == self.city_var.get()]
                all_firing_id = city_data['firingID'].value_counts()[city_data['firingID'].value_counts() >= 2].index


            firing_id_numbers = list(all_firing_id)
            firing_id_numbers.insert(0, "All")

            patrol_id_dropdown = ttk.Combobox(frame, values=firing_id_numbers, width=25, textvariable=self.firing_id_var)
            patrol_id_dropdown.current(0)
            patrol_id_dropdown.bind("<<ComboboxSelected>>", self.prepare_data)
            patrol_id_dropdown.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        else:
            self.firing_id_var.set('All')

            # Tablica posiadająca wszystkie district_name
            if self.city_var.get() == 'All':
                all_district_name = self.data['districtName'].unique()
            else:
                city_data = self.data[self.data['City'] == self.city_var.get()]
                all_district_name = city_data['districtName'].unique()

            district_name_numbers = list(all_district_name)
            district_name_numbers.insert(0, "All")

            distinct_dropdown = ttk.Combobox(frame, values=district_name_numbers, width=28, textvariable=self.distinct_var)
            distinct_dropdown.current(0)
            distinct_dropdown.bind("<<ComboboxSelected>>", self.prepare_data)
            distinct_dropdown.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

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
        filtered_df = self.data.copy()
        filtered_df["arrivedPatrols"] = filtered_df["solvingPatrols"] + filtered_df["reachingPatrols(including 'called')"]


        if self.city_var.get() != '' and self.firing_id_var.get() != '' and self.distinct_var.get() != '': 
            if self.city_var.get() == 'All':
                if self.firing_id_var.get() != 'All':
                    filtered_df = filtered_df[filtered_df["firingID"] == self.firing_id_var.get()]
                if self.distinct_var.get() != "All":
                    filtered_df = filtered_df[filtered_df["districtName"] == self.distinct_var.get()]
            else:
                if self.firing_id_var.get() != 'All':
                    filtered_df = filtered_df[(filtered_df["firingID"] == self.firing_id_var.get()) & (self.data['City'] == self.city_var.get())]
                if self.distinct_var.get() != "All":
                    filtered_df = filtered_df[(filtered_df["districtName"] == self.distinct_var.get()) & (self.data['City'] == self.city_var.get())]
                if self.firing_id_var.get() == 'All' and self.distinct_var.get() == "All":
                    filtered_df = filtered_df[(self.data['City'] == self.city_var.get())]

        filtered_df.sort_values(by="simulationTime[s]", inplace=True)
        filtered_df["simulationTime[h]"] = filtered_df["simulationTime[s]"] / 3600
        filtered_df["totalDistanceOfCalledPatrols"] = filtered_df["totalDistanceOfCalledPatrols"].replace({',': '.'}, regex=True).astype(float) / 1000

        df = filtered_df.copy()

        # Wybór interesujących kolumn
        selected_columns = [
            "generallyRequiredPatrols",
            "arrivedPatrols",
            "solvingPatrols",
            "reachingPatrols(including 'called')",
            "calledPatrols"
        ]

        # Tworzenie DataFrame
        grouped_data = pd.DataFrame(columns=['Operation'] + selected_columns)

        # Dodanie pierwszego wiersza z sumą
        grouped_data.loc[0] = ['sum'] + df[selected_columns].sum().tolist()
        grouped_data.loc[1] = ['mean'] + df[selected_columns].mean().tolist()

        # Dodanie wiersza z danymi, gdy generallyRequiredPatrols jest największe
        max_row_index = df['generallyRequiredPatrols'].idxmax()
        max_row = df.loc[max_row_index, selected_columns].tolist()
        grouped_data.loc[2] = ['max of required'] + max_row

        # Dodanie wiersza z danymi, gdy generallyRequiredPatrols jest najmniejsze
        min_row_index = df['generallyRequiredPatrols'].idxmin()
        min_row = df.loc[min_row_index, selected_columns].tolist()
        grouped_data.loc[3] = ['min of required'] + min_row

        # Dodanie nowej kolumny z wynikiem działania
        grouped_data['Difference_patrols'] = grouped_data['generallyRequiredPatrols'] - (grouped_data['solvingPatrols'] + grouped_data['reachingPatrols(including \'called\')'])

        if self.presentation_mode == 'chart':
            self.draw_chart(filtered_df)
            self.analyse_for_values(grouped_data, selected_columns)
        elif self.presentation_mode == 'table':
            self.create_table(filtered_df)

        # Utworzenie buttona do exportu danych
        button = ttk.Button(self.export_frame, text="Export data", command=lambda: export_to_csv(filtered_df, f"Firings Details - {self.city_var.get()} - {self.distinct_var.get()}"))
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

        # Sporządzenie wykresów
        if self.mode_dropdown_var.get() == "Mean":
            mean_data = filtered_df[self.columns_to_plot]

            if self.distinct_var.get() != "All":
                mean_data = mean_data[filtered_df["districtName"] == self.distinct_var.get()]
            if self.firing_id_var.get() != "All":
                mean_data = mean_data[filtered_df["firingID"] == self.firing_id_var.get()]

            mean_data = mean_data.mean()

            # Stworzenie wykresu za pomocą seaborn
            ax = sns.barplot(x=mean_data.values, y=mean_data.index, hue=mean_data.index, palette='viridis')

            plt.subplots_adjust(left=0.2, right=0.95)

            # Dodanie etykiet do słupków
            for index, value in enumerate(mean_data.values):
                plt.text(value, index, f'{value:.2f}', ha='left', va='center', color='white')

            # Ustawienia tytułu i osi
            if self.distinct_var.get() != 'All' and self.firing_id_var.get() == 'All':
                if self.city_var.get() != 'All':
                    title = (f'for {self.distinct_var.get()} in {self.city_var.get()}')
                else:
                    title = (f'for {self.distinct_var.get()}')
            elif self.distinct_var.get() == 'All' and self.firing_id_var.get() != 'All':
                if self.city_var.get() != 'All':
                    title = (f'for patrol {self.firing_id_var.get()} in {self.city_var.get()}')
                else:
                    title = (f'for patrol {self.firing_id_var.get()}')
            else:
                if self.city_var.get() != 'All':
                    title = (f'for {self.city_var.get()}')
                else:
                    title = ""

            plt.title(f"Mean Patrols Value {title}")
            plt.xlabel("Mean Value")
            plt.ylabel("Patrols Category")
        elif self.mode_dropdown_var.get() == "Amount":
            mean_data = filtered_df[self.columns_to_plot]

            if self.distinct_var.get() != "All":
                mean_data = mean_data[filtered_df["districtName"] == self.distinct_var.get()]
            if self.firing_id_var.get() != "All":
                mean_data = mean_data[filtered_df["firingID"] == self.firing_id_var.get()]

            mean_data = mean_data.sum()

            # Stworzenie wykresu za pomocą seaborn
            ax = sns.barplot(x=mean_data.values, y=mean_data.index, hue=mean_data.index, palette='viridis')

            plt.subplots_adjust(left=0.2, right=0.95)

            # Dodanie etykiet do słupków
            for index, value in enumerate(mean_data.values):
                plt.text(value, index, f'{value:.2f}', ha='left', va='center', color='white')

            # Ustawienia tytułu i osi
            if self.distinct_var.get() != 'All' and self.firing_id_var.get() == 'All':
                if self.city_var.get() != 'All':
                    title = (f'for {self.distinct_var.get()} in {self.city_var.get()}')
                else:
                    title = (f'for {self.distinct_var.get()}')
            elif self.distinct_var.get() == 'All' and self.firing_id_var.get() != 'All':
                if self.city_var.get() != 'All':
                    title = (f'for patrol {self.firing_id_var.get()} in {self.city_var.get()}')
                else:
                    title = (f'for patrol {self.firing_id_var.get()}')
            else:
                if self.city_var.get() != 'All':
                    title = (f'for {self.city_var.get()}')
                else:
                    title = ""

            plt.title(f"Amount Patrols Value {title}")
            plt.xlabel("Amount Value")
            plt.ylabel("Patrols Category")
        

        container_frame = ttk.Frame(self.frame_chart)
        container_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Osadzenie wykresu w interfejsie tkinter i narysowanie
        canvas = FigureCanvasTkAgg(fig, master=container_frame)
        canvas.draw()

        # Utworzenie buttona do exportu wykresu
        button = ttk.Button(self.export_frame, text="Export chart", command=lambda: export_plot_to_image(fig, f"Redundancy - {self.city_var.get()} - {self.distinct_var.get()}"))
        button.grid(row=1, column=0, padx=15, pady=5, sticky="nsew")

        # Zamyknięcie wykresu po użyciu
        plt.close(fig)

        # Dodanie paseka narzędziowego
        toolbar = NavigationToolbar2Tk(canvas, container_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Umieszczenie canvas w grid w frame
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


    # Analiza danych dla Values i Mean
    def analyse_for_values(self, data, columns):        
        # Wyczyszczenie frame
        for widget in self.frame_analise.winfo_children():
            widget.destroy()

        # Utwórzenie drzewa do wyświetlania danych
        tree = ttk.Treeview(self.frame_analise, columns=['Operation'] + columns + ['Difference_patrols'], show="headings")

        # Dodanie kolumny
        tree.heading("Operation", text="Operation")
        tree.heading("generallyRequiredPatrols", text="Required Patrols")
        tree.heading("arrivedPatrols", text="Arrived Patrols")
        tree.heading("solvingPatrols", text="Solving patrols")
        tree.heading("reachingPatrols(including 'called')", text="Reaching patrols")
        tree.heading("calledPatrols", text="Called Patrols")
        tree.heading("Difference_patrols", text="Difference patrols")

        tree.column("#1", anchor="center", width=160)
        tree.column("#2", anchor="center", width=160)
        tree.column("#3", anchor="center", width=160)
        tree.column("#4", anchor="center", width=160)
        tree.column("#5", anchor="center", width=160)
        tree.column("#6", anchor="center", width=160)
        tree.column("#6", anchor="center", width=160)
        tree.column("#7", anchor="center", width=160)

        for index, row in data.iterrows():
            tree.insert("", "end", values=(row['Operation'], f"{row['generallyRequiredPatrols']:.2f}", f"{row['arrivedPatrols']:.2f}", f"{row['solvingPatrols']:.2f}",
                                            f"{row.iloc[4]:.2f}",  f"{row['calledPatrols']:.2f}", f"{row['Difference_patrols']:.2f}"))

        tree.pack(side="left", fill="both", expand=True)

        ###########################
        ########## WYKRES #########
        ###########################
        # Rysowanie histogramu
        fig, ax = plt.subplots(figsize=(10, 4))

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

        if self.mode_dropdown_var.get() == "Amount":
            # Przygotowanie danych
            categories = ["Required", "Patrols", "Difference"]
            # values = data.drop('Operation', axis=1).iloc[1, :]
            values = data.iloc[1,1], data.iloc[1,2] + data.iloc[1,3], data.iloc[1,5]

            ax.bar(categories, values)
            ax.set_xlabel('Operation')
            ax.set_ylabel('Value')
            ax.set_title('Mean Values of Operations')

        elif self.mode_dropdown_var.get() == "Mean":
            # Przygotowanie danych
            # categories = data['Operation']
            categories = ["Required", "Patrols", "Difference"]
            # values = data.drop('Operation', axis=1).iloc[1, :]
            values = data.iloc[0,1], data.iloc[0,2] + data.iloc[0,3], data.iloc[0,5]

            ax.bar(categories, values)
            ax.set_xlabel('Operation')
            ax.set_ylabel('Value')
            ax.set_title('Amount Values of Operations')          

        fig.subplots_adjust(left=0.25, right=0.95, bottom=0.25, top=0.85)

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