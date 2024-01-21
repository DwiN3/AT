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


class AmbulancesAndSwatVisualizer():
    def __init__(self, frame1, frame2, selected_cities):
        super().__init__()

        self.frame1 = frame1
        self.frame2 = frame2

        self.data_Avg_Swat = load_data(selected_cities, "Average Swat Distance And Time To Reach Firing")

        self.data_Avg_Ambulances = load_data(selected_cities, "Average Ambulance Distance And Time To Reach Firing")

        self.data_Used_Swat = load_data(selected_cities, "Used Swat Per Hour")

        self.data_Used_Ambulances = load_data(selected_cities, "Ambulances In Use Per Hour")

        self.selected_cities = list(selected_cities)

        ###########################
        ######### ZMIENNE #########
        ###########################
        self.city_var = tk.StringVar()
        self.mode_dropdown_var = tk.StringVar()

        self.city_state_radios = []
        self.presentation_mode = 'chart'

        self.modes = [
            'Average Distance To Reach Firing',
            'Average Time To Reach Firing',
            'Used Per Hour'
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

        for i, city in enumerate(self.selected_cities):
            mode_radio = ttk.Radiobutton(set_city_frame, text=city, value=city, variable=self.city_var, command=self.prepare_data)
            mode_radio.grid(row=i , column=0, padx=5, pady=5, sticky="nsew")
        self.city_var.set(self.selected_cities[0])

        set_mode_frame = ttk.LabelFrame(self.options_frame, text="Set frame mode")
        set_mode_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        mode_dropdown = ttk.Combobox(set_mode_frame, values=self.modes, width=28, textvariable=self.mode_dropdown_var)
        mode_dropdown.current(0)
        mode_dropdown.bind("<<ComboboxSelected>>", self.prepare_data)
        mode_dropdown.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

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
        if self.mode_dropdown_var.get() == 'Average Distance To Reach Firing' or self.mode_dropdown_var.get() == 'Average Time To Reach Firing':
            df_swat = self.data_Avg_Swat[self.data_Avg_Ambulances['City'] == self.city_var.get()]
            df_swat = df_swat.copy()
            df_swat["simulationTime[h]"] = df_swat["simulationTime[s]"] / 3600
            df_swat["averageTimeToReach[min]"] = df_swat["averageTimeToReach[s]"] / 60
            df_swat["averageDistanceToReach[km]"] = df_swat["averageDistanceToReach[m]"] / 1000
            
            
            df_ambulance = self.data_Avg_Ambulances[self.data_Avg_Ambulances['City'] == self.city_var.get()]
            df_ambulance = df_ambulance.copy()
            df_ambulance["simulationTime[h]"] = df_ambulance["simulationTime[s]"] / 3600
            df_ambulance["averageTimeToReach[min]"] = df_ambulance["averageTimeToReach[s]"] / 60
            df_ambulance["averageDistanceToReach[km]"] = df_ambulance["averageDistanceToReach[m]"] / 1000

            selected_columns = [
                'averageTimeToReach[min]',
                'averageDistanceToReach[km]',
            ]

            df_to_analise = pd.DataFrame(columns=['Operation'] + selected_columns)
            
            sum_row_swat = df_swat[selected_columns].sum().tolist()
            df_to_analise.loc[0] = ['Sum for Swat'] +  sum_row_swat

            mean_row_swat = df_swat[selected_columns].mean().tolist()
            df_to_analise.loc[1] = ['Average for Swat'] +  mean_row_swat

            max_row__index_swat = df_swat['averageTimeToReach[min]'].idxmax()
            max_row_swat = df_swat.loc[max_row__index_swat, selected_columns].tolist()
            df_to_analise.loc[2] = ['Max values for Swat'] +  max_row_swat

            min_row__index_swat = df_swat[df_swat['averageTimeToReach[min]'] > 0]['averageTimeToReach[min]'].idxmin() 
            min_row_swat = df_swat.loc[min_row__index_swat, selected_columns].tolist()
            df_to_analise.loc[3] = ['Min values for Swat'] +  min_row_swat

            sum_row_ambulance = df_ambulance[selected_columns].sum().tolist()
            df_to_analise.loc[4] = ['Sum for Ambulance'] +  sum_row_ambulance

            mean_row_ambulance = df_ambulance[selected_columns].mean().tolist()
            df_to_analise.loc[5] = ['Average for Ambulance'] +  mean_row_ambulance

            max_row__index_ambulance = df_ambulance['averageTimeToReach[min]'].idxmax()
            max_row_ambulance = df_ambulance.loc[max_row__index_ambulance, selected_columns].tolist()
            df_to_analise.loc[6] = ['Max values for Ambulance'] +  max_row_ambulance

            min_row__index_ambulance = df_ambulance[df_ambulance['averageTimeToReach[min]'] > 0]['averageTimeToReach[min]'].idxmin() 
            min_row_ambulance = df_ambulance.loc[min_row__index_ambulance, selected_columns].tolist()
            df_to_analise.loc[7] = ['Min values for Swat'] +  min_row_ambulance

            # Dodanie kategorii do DataFrame'ów przed połączeniem
            df_swat['Category'] = 'Swat'
            df_ambulance['Category'] = 'Ambulance'

            # Połączenie ramek danych
            df_data = pd.concat([df_swat, df_ambulance])

            # Wybór interesujących kolumn
            selected_columns = ['simulationTime[s]', 'averageTimeToReach[s]', 'averageDistanceToReach[m]', 'Category', 'City']
            df_data = df_data[selected_columns]

            # Sortowanie po simulationTime[s] rosnąco
            df_data = df_data.sort_values(by='simulationTime[s]').reset_index(drop=True)


            if self.presentation_mode == 'chart':
                self.draw_chart(df_swat, df_ambulance)
                self.analyse(df_to_analise)
            elif self.presentation_mode == 'table':
                self.create_table(df_data)

            # Utworzenie buttona do exportu danych
            button = ttk.Button(self.export_frame, text="Export data", command=lambda: export_to_csv(df_data, f"Comparison Of Ambulances With Swat - {self.city_var.get()}"))
            button.grid(row=0, column=0, padx=15, pady=5, sticky="nsew")
        

        elif self.mode_dropdown_var.get() == 'Used Per Hour':
            
            df_Swat = self.data_Used_Swat[self.data_Avg_Ambulances['City'] == self.city_var.get()]
            df_Swat = df_Swat.copy()
            df_Swat["simulationTime[h]"] = df_Swat["simulationTime[s]"] / 3600

            df_Ambulance = self.data_Used_Ambulances[self.data_Avg_Ambulances['City'] == self.city_var.get()]
            df_Ambulance = df_Ambulance.copy()
            df_Ambulance["simulationTime[h]"] = df_Ambulance["simulationTime[s]"] / 3600

            df_to_Analise = pd.DataFrame(columns=['Operation'] + ['Value'])
        
            sum_row_swat = df_Swat['amountOfUsedSwat'].sum()
            df_to_Analise.loc[0] = ['Sum for Swat', sum_row_swat]

            mean_row_swat = df_Swat['amountOfUsedSwat'].mean()
            df_to_Analise.loc[1] = ['Average for Swat',  mean_row_swat]

            max_row__index_swat = df_Swat['amountOfUsedSwat'].idxmax()
            max_row_swat = df_Swat.loc[max_row__index_swat, 'amountOfUsedSwat']
            df_to_Analise.loc[2] = ['Max values for Swat',  max_row_swat]

            min_row__index_swat = df_Swat[df_Swat['amountOfUsedSwat'] > 0]['amountOfUsedSwat'].idxmin() 
            min_row_swat = df_Swat.loc[min_row__index_swat, 'amountOfUsedSwat']
            df_to_Analise.loc[3] = ['Min values for Swat', min_row_swat]

            sum_row_ambulance = df_Ambulance['amountOfSolvingAmbulances'].sum()
            df_to_Analise.loc[4] = ['Sum for Ambulance', sum_row_ambulance]

            mean_row_ambulance = df_Ambulance['amountOfSolvingAmbulances'].mean()
            df_to_Analise.loc[5] = ['Average for Ambulance', mean_row_ambulance]

            max_row__index_ambulance = df_Ambulance['amountOfSolvingAmbulances'].idxmax()
            max_row_ambulance = df_Ambulance.loc[max_row__index_ambulance, 'amountOfSolvingAmbulances']
            df_to_Analise.loc[6] = ['Max values for Ambulance', max_row_ambulance]

            min_row__index_ambulance = df_Ambulance[df_Ambulance['amountOfSolvingAmbulances'] > 0]['amountOfSolvingAmbulances'].idxmin() 
            min_row_ambulance = df_Ambulance.loc[min_row__index_ambulance, 'amountOfSolvingAmbulances']
            df_to_Analise.loc[7] = ['Min values for Swat', min_row_ambulance]

            # Dodanie kategorii do DataFrame'ów przed połączeniem
            df_Swat['Category'] = 'Swat'
            df_Swat['Value'] = df_Swat['amountOfUsedSwat']
            df_Ambulance['Category'] = 'Ambulance'
            df_Ambulance['Value'] = df_Ambulance['amountOfSolvingAmbulances']

            # Połączenie ramek danych
            df_Data = pd.concat([df_Swat, df_Ambulance])

            # Wybór interesujących kolumn
            selected_columns = ['simulationTime[s]', 'Value', 'Category', 'City']
            df_Data = df_Data[selected_columns]

            # Sortowanie po simulationTime[s] rosnąco
            df_Data = df_Data.sort_values(by='simulationTime[s]').reset_index(drop=True)


            if self.presentation_mode == 'chart':
                self.draw_chart(df_Swat, df_Ambulance)
                self.analyse(df_to_Analise)
            elif self.presentation_mode == 'table':
                self.create_table(df_Data)


            # Utworzenie buttona do exportu danych
            button = ttk.Button(self.export_frame, text="Export data", command=lambda: export_to_csv(df_Data, f"Comparison Of Ambulances With Swat - {self.city_var.get()}"))
            button.grid(row=0, column=0, padx=15, pady=5, sticky="nsew")


    # Rysowanie wykresu
    def draw_chart(self, df_swat, df_ambulance):
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
        
        if self.mode_dropdown_var.get() == 'Average Distance To Reach Firing': 
            # Narysowanie wykresu
            plt.plot(df_swat["simulationTime[h]"], df_swat["averageDistanceToReach[km]"], label="Average Distance Swat")    
            plt.plot(df_ambulance["simulationTime[h]"], df_ambulance["averageDistanceToReach[km]"], label="Average Distance Ambulance")
            # Dodanie etykiet oraz tytułu wykresu
            plt.title(f"Average Distance To Reach Firing for {self.city_var.get()}")
            plt.xlabel("Simulation Time [h]")
            plt.ylabel("Average Distance [km]")
        elif self.mode_dropdown_var.get() == 'Average Time To Reach Firing':
            # Narysowanie wykresu              
            plt.plot(df_swat["simulationTime[h]"], df_swat["averageTimeToReach[min]"], label="Average Time Swat")
            plt.plot(df_ambulance["simulationTime[h]"], df_ambulance["averageTimeToReach[min]"], label="Average Time Ambulance")
            # Dodanie etykiet oraz tytułu wykresu
            plt.title(f"Average Time To Reach Firing for {self.city_var.get()}")
            plt.xlabel("Simulation Time [h]")
            plt.ylabel("Average Time [min]")
        elif self.mode_dropdown_var.get() == 'Used Per Hour':
            # Narysowanie wykresu
            plt.plot(df_swat["simulationTime[h]"], df_swat["amountOfUsedSwat"], label="Amount of Swat")
            plt.plot(df_ambulance["simulationTime[h]"], df_ambulance["amountOfSolvingAmbulances"], label="Amount of Ambulance")
            # Dodanie etykiet oraz tytułu wykresu
            plt.title(f"Ambulance and Swat ussage per hour for {self.city_var.get()}")
            plt.xlabel("Simulation Time [h]")
            plt.ylabel("Amount")

        # Dodanie legendy
        plt.legend()

        # Ustawienie przyrostów na osiach
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(1))
        # ax.yaxis.set_major_locator(MultipleLocator(5))

        container_frame = ttk.Frame(self.frame_chart)
        container_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Osadzenie wykresu w interfejsie tkinter i narysowanie
        canvas = FigureCanvasTkAgg(fig, master=container_frame)
        canvas.draw()

        # Utworzenie buttona do exportu wykresu
        button = ttk.Button(self.export_frame, text="Export chart", command=lambda: export_plot_to_image(fig, f"Comparison Of Ambulances With Swat - {self.city_var.get()}"))
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

        if self.mode_dropdown_var.get() == 'Average Distance To Reach Firing' or self.mode_dropdown_var.get() == 'Average Time To Reach Firing':
            # Utwórzenie drzewa do wyświetlania danych
            tree = ttk.Treeview(self.frame_analise,
                                columns=["Option", "averageTimeToReach", "averageDistanceToReach"], show="headings")

            # Dodanie kolumny
            tree.heading("Option", text="Operation")
            tree.heading("averageTimeToReach", text="Average Time To Reach [min]")
            tree.heading("averageDistanceToReach", text="Average Distance To Reach [km]")

            tree.column("#1", anchor="center")
            tree.column("#2", anchor="center")
            tree.column("#3", anchor="center")

            for index, row in data.iterrows():
                tree.insert("", "end", values=(row['Operation'], f"{row['averageTimeToReach[min]']:.2f}min", f"{row['averageDistanceToReach[km]']:.3f}km"))

            tree.pack(side="left", fill="both", expand=True)

            ###########################
            ########## WYKRES #########
            ###########################
            # Rysowanie histogramu
            fig, ax = plt.subplots(figsize=(6, 3))

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

            # Wybrane operacje
            selected_operations = ['Average for Swat', 'Average for Ambulance']

            # Filtruj DataFrame według wybranych operacji
            filtered_data = data[data['Operation'].isin(selected_operations)]

            # Rysowanie słupków dla czasu i dystansu
            bar_width = 0.35
            bar1 = ax.bar(filtered_data['Operation'], filtered_data['averageTimeToReach[min]'], bar_width, label='Time [min]')
            bar2 = ax.bar(filtered_data['Operation'], filtered_data['averageDistanceToReach[km]'], bar_width, label='Distance [km]', bottom=filtered_data['averageTimeToReach[min]'])

            # Ustawienia osi i etykiet
            ax.set_xlabel('Operation')
            ax.set_ylabel('Value')
            ax.set_title('Comparison of Average Time and Distance for Swat and Ambulance')
            ax.legend()

            # Dodanie wartości numerycznych nad słupkami
            def add_values_labels(bars):
                for bar in bars:
                    yval = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

            add_values_labels(bar1)
            add_values_labels(bar2)

            fig.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.90)

            # Osadzenie wykresu w interfejsie tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.frame_analise)
            canvas.draw()

            # Zamyknięcie wykresu po użyciu
            plt.close(fig)

            canvas.get_tk_widget().pack(side=tk.LEFT, padx=10, pady=10)

        elif self.mode_dropdown_var.get() == 'Used Per Hour':
            # Utwórzenie drzewa do wyświetlania danych
            tree = ttk.Treeview(self.frame_analise, columns=["Option", "Value"], show="headings")

            # Dodanie kolumny
            tree.heading("Option", text="Operation")
            tree.heading("Value", text="Value")

            tree.column("#1", anchor="center")
            tree.column("#2", anchor="center")

            for index, row in data.iterrows():
                tree.insert("", "end", values=(row['Operation'], f"{row['Value']:.2f}"))

            tree.pack(side="left", fill="both", expand=True)

            ###########################
            ########## WYKRES #########
            ###########################
            # Rysowanie histogramu
            fig, ax = plt.subplots(figsize=(6, 3))

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

            sns.barplot(x='Operation', y='Value', hue='Operation', data=data)
            plt.title('Comparison of Swat and Ambulance')
            plt.xticks(rotation=45, ha='right')

            fig.subplots_adjust(left=0.10, right=0.95, bottom=0.50, top=0.90)

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