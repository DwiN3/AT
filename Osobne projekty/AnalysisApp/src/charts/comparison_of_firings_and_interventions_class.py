import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from src.dataOperations.export_chart import export_plot_to_image
import pandas as pd
from src.dataOperations.export_data import export_to_csv
from src.dataOperations.load_data import load_data


class InterventionAndFiringsVisualizer():
    def __init__(self, frame1, frame2, selected_cities):
        super().__init__()

        self.frame1 = frame1
        self.frame2 = frame2

        self.data_duration = load_data(selected_cities, "Average Duration Of Incidents Per Hour")

        self.data_transfer = load_data(selected_cities, "Average Duration Patrols Heading Towards Incidents Per Hour")

        self.selected_cities = list(selected_cities)

        ###########################
        ######### ZMIENNE #########
        ###########################
        self.city_var = tk.StringVar()
        self.mode_dropdown_var = tk.StringVar()
        self.data_type_dropdown_var = tk.StringVar()

        self.city_state_radios = []
        self.presentation_mode = 'chart'

        self.data_type = [
            "Firings and Interventions",
            "Firings",
            "Interventions"
        ]

        self.modes = [
            "Amount",
            "Duration",
            "Average Duration",
            "Average Transfer",
            "Incidents vs Transfer Time"
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

        set_mode_frame = ttk.LabelFrame(self.options_frame, text="Set data type")
        set_mode_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        mode_dropdown = ttk.Combobox(set_mode_frame, values=self.data_type, width=28, textvariable=self.data_type_dropdown_var)
        mode_dropdown.current(0)
        mode_dropdown.bind("<<ComboboxSelected>>", self.prepare_data)
        mode_dropdown.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        set_mode_frame = ttk.LabelFrame(self.options_frame, text="Set mode")
        set_mode_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")

        mode_dropdown = ttk.Combobox(set_mode_frame, values=self.modes, width=28, textvariable=self.mode_dropdown_var)
        mode_dropdown.current(0)
        mode_dropdown.bind("<<ComboboxSelected>>", self.prepare_data)
        mode_dropdown.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.prepare_data()

        display_frame = ttk.LabelFrame(self.options_frame, text="Display")
        display_frame.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")

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

        df_transfer = self.data_transfer[self.data_transfer['City'] == self.city_var.get()]
        df_transfer = df_transfer.copy()
        df_transfer['averageTransferToInterventionDuration[min]'] = df_transfer["averageTransferToInterventionDuration[s]"] / 60
        df_transfer['averageTransferToFiringDuration[min]'] = df_transfer['averageTransferToFiringDuration[s]'] / 60
        df_transfer['averageTransferToIncidents[min]'] = df_transfer['averageTransferToInterventionDuration[min]'] + df_transfer['averageTransferToFiringDuration[min]']

        df_duration = self.data_duration[self.data_duration['City'] == self.city_var.get()]
        df_duration = df_duration.copy()
        df_duration['firingsDuration[h]'] = df_duration['firingsDuration[min]'] / 60
        df_duration['interventionsDuration[h]'] = df_duration['interventionsDuration[min]'] / 60
        df_duration['averageFiringDuration[h]'] = df_duration['averageFiringDuration[min]'] / 60
        df_duration['averageInterventionDuration[h]'] = df_duration['averageInterventionDuration[min]'] / 60
        df_duration['amountOfIncidents'] = df_duration['amountOfInterventions'] + df_duration['amountOfFirings']

        df_to_analise = pd.DataFrame(columns=['Operation'] + ['Value'])

        if self.mode_dropdown_var.get() == "Amount" or self.mode_dropdown_var.get() == "Duration" or self.mode_dropdown_var.get() == "Average Duration":
            if self.mode_dropdown_var.get() == "Amount":
                intervention_name = 'amountOfInterventions'
                firing_name = 'amountOfFirings'
            elif self.mode_dropdown_var.get() == "Duration":
                intervention_name = 'interventionsDuration[h]'
                firing_name = 'firingsDuration[h]'
            elif self.mode_dropdown_var.get() == "Average Duration":
                intervention_name = 'averageInterventionDuration[min]'
                firing_name = 'averageFiringDuration[min]'

            if self.data_type_dropdown_var.get() == "Firings and Interventions":

                sum_row_interventions = df_duration[intervention_name].sum()
                df_to_analise.loc[0] = ['Sum of Interventions', sum_row_interventions]

                mean_row_interventions = df_duration[intervention_name].mean()
                df_to_analise.loc[1] = ['Average of Interventions per Hour', mean_row_interventions]

                max_row__index_intervention = df_duration[intervention_name].idxmax()
                max_row_intervention = df_duration.loc[max_row__index_intervention, intervention_name].tolist()
                df_to_analise.loc[2] = ['Max of Interventions per Hour', max_row_intervention]

                min_row__index_intervention = df_duration[intervention_name].idxmin() 
                min_row_intervention = df_duration.loc[min_row__index_intervention, intervention_name].tolist()
                df_to_analise.loc[3] = ['Min of Interventions per Hour', min_row_intervention]

                sum_row_firings = df_duration[firing_name].sum()
                df_to_analise.loc[4] = ['Sum of Firings', sum_row_firings]

                mean_row_firings = df_duration[firing_name].mean()
                df_to_analise.loc[5] = ['Average of Firings per Hour', mean_row_firings]

                max_row__index_firing = df_duration[firing_name].idxmax()
                max_row_firing = df_duration.loc[max_row__index_firing, firing_name].tolist()
                df_to_analise.loc[6] = ['Max of Firings per Hour', max_row_firing]

                min_row__index_firing = df_duration[firing_name].idxmin() 
                min_row_firing = df_duration.loc[min_row__index_firing, firing_name].tolist()
                df_to_analise.loc[7] = ['Min of Firings per Hour', min_row_firing]

            if self.data_type_dropdown_var.get() == "Firings":
                sum_row_firings = df_duration[firing_name].sum()
                df_to_analise.loc[0] = ['Sum of Firings', sum_row_firings]

                mean_row_firings = df_duration[firing_name].mean()
                df_to_analise.loc[1] = ['Average of Firings per Hour', mean_row_firings]

                max_row__index_firing = df_duration[firing_name].idxmax()
                max_row_firing = df_duration.loc[max_row__index_firing, firing_name].tolist()
                df_to_analise.loc[2] = ['Max of Firings per Hour', max_row_firing]

                min_row__index_firing = df_duration[firing_name].idxmin() 
                min_row_firing = df_duration.loc[min_row__index_firing, firing_name].tolist()
                df_to_analise.loc[3] = ['Min of Firings per Hour', min_row_firing]

            if self.data_type_dropdown_var.get() == "Interventions":

                sum_row_interventions = df_duration[intervention_name].sum()
                df_to_analise.loc[0] = ['Sum of Interventions', sum_row_interventions]

                mean_row_interventions = df_duration[intervention_name].mean()
                df_to_analise.loc[1] = ['Average of Interventions per Hour', mean_row_interventions]

                max_row__index_intervention = df_duration[intervention_name].idxmax()
                max_row_intervention = df_duration.loc[max_row__index_intervention, intervention_name].tolist()
                df_to_analise.loc[2] = ['Max of Interventions per Hour', max_row_intervention]

                min_row__index_intervention = df_duration[intervention_name].idxmin() 
                min_row_intervention = df_duration.loc[min_row__index_intervention, intervention_name].tolist()
                df_to_analise.loc[3] = ['Min of Interventions per Hour', min_row_intervention]
        
        elif self.mode_dropdown_var.get() == "Average Transfer":
            intervention_name = 'averageTransferToInterventionDuration[min]'
            firing_name = 'averageTransferToFiringDuration[min]'

            if self.data_type_dropdown_var.get() == "Firings and Interventions":

                sum_row_interventions = df_transfer[intervention_name].sum()
                df_to_analise.loc[0] = ['Sum Of Transfer Time To Interventions', sum_row_interventions]

                mean_row_interventions = df_transfer[intervention_name].mean()
                df_to_analise.loc[1] = ['Average Of Transfer Time To Interventions per Hour', mean_row_interventions]

                max_row__index_intervention = df_transfer[intervention_name].idxmax()
                max_row_intervention = df_transfer.loc[max_row__index_intervention, intervention_name].tolist()
                df_to_analise.loc[2] = ['Max Of Transfer Time To Interventions per Hour', max_row_intervention]

                min_row__index_intervention = df_transfer[intervention_name].idxmin() 
                min_row_intervention = df_transfer.loc[min_row__index_intervention, intervention_name].tolist()
                df_to_analise.loc[3] = ['Min Of Transfer Time To Interventions per Hour', min_row_intervention]

                sum_row_firings = df_transfer[firing_name].sum()
                df_to_analise.loc[4] = ['Sum Of Transfer Time To Firings', sum_row_firings]

                mean_row_firings = df_transfer[firing_name].mean()
                df_to_analise.loc[5] = ['Average Of Transfer Time To Firings per Hour', mean_row_firings]

                max_row__index_firing = df_transfer[firing_name].idxmax()
                max_row_firing = df_transfer.loc[max_row__index_firing, firing_name].tolist()
                df_to_analise.loc[6] = ['Max Of Transfer Time To Firings per Hour', max_row_firing]

                min_row__index_firing = df_transfer[firing_name].idxmin() 
                min_row_firing = df_transfer.loc[min_row__index_firing, firing_name].tolist()
                df_to_analise.loc[7] = ['Min Of Transfer Time To Firings per Hour', min_row_firing]

            if self.data_type_dropdown_var.get() == "Firings":
                sum_row_firings = df_transfer[firing_name].sum()
                df_to_analise.loc[0] = ['Sum Of Transfer Time To Firings', sum_row_firings]

                mean_row_firings = df_transfer[firing_name].mean()
                df_to_analise.loc[1] = ['Average Of Transfer Time To Firings per Hour', mean_row_firings]

                max_row__index_firing = df_transfer[firing_name].idxmax()
                max_row_firing = df_transfer.loc[max_row__index_firing, firing_name].tolist()
                df_to_analise.loc[2] = ['Max Of Transfer Time To Firings per Hour', max_row_firing]

                min_row__index_firing = df_transfer[firing_name].idxmin() 
                min_row_firing = df_transfer.loc[min_row__index_firing, firing_name].tolist()
                df_to_analise.loc[3] = ['Min Of Transfer Time To Firings per Hour', min_row_firing]

            if self.data_type_dropdown_var.get() == "Interventions":

                sum_row_interventions = df_transfer[intervention_name].sum()
                df_to_analise.loc[0] = ['Sum Of Transfer Time To Interventions', sum_row_interventions]

                mean_row_interventions = df_transfer[intervention_name].mean()
                df_to_analise.loc[1] = ['Average Of Transfer Time To Interventions per Hour', mean_row_interventions]

                max_row__index_intervention = df_transfer[intervention_name].idxmax()
                max_row_intervention = df_transfer.loc[max_row__index_intervention, intervention_name].tolist()
                df_to_analise.loc[2] = ['Max Of Transfer Time To Interventions per Hour', max_row_intervention]

                min_row__index_intervention = df_transfer[intervention_name].idxmin() 
                min_row_intervention = df_transfer.loc[min_row__index_intervention, intervention_name].tolist()
                df_to_analise.loc[3] = ['Min Of Transfer Time To Interventions per Hour', min_row_intervention]

        elif self.mode_dropdown_var.get() == "Incidents vs Transfer Time":

            if self.data_type_dropdown_var.get() == "Firings and Interventions":            
                intervention_name = 'amountOfIncidents'
                firing_name = 'averageTransferToIncidents[min]'

            if self.data_type_dropdown_var.get() == "Firings":
                intervention_name = 'amountOfInterventions'
                firing_name = 'averageTransferToFiringDuration[min]'

            if self.data_type_dropdown_var.get() == "Interventions":
                intervention_name = 'amountOfFirings'
                firing_name = 'averageTransferToFiringDuration[min]'

            sum_row_interventions = df_duration[intervention_name].sum()
            df_to_analise.loc[0] = ['Sum of Incidents', sum_row_interventions]

            mean_row_interventions = df_duration[intervention_name].mean()
            df_to_analise.loc[1] = ['Average of Incidents per Hour', mean_row_interventions]

            max_row__index_intervention = df_duration[intervention_name].idxmax()
            max_row_intervention = df_duration.loc[max_row__index_intervention, intervention_name].tolist()
            df_to_analise.loc[2] = ['Max of Incidents per Hour', max_row_intervention]

            min_row__index_intervention = df_duration[intervention_name].idxmin() 
            min_row_intervention = df_duration.loc[min_row__index_intervention, intervention_name].tolist()
            df_to_analise.loc[3] = ['Min of Incidents per Hour', min_row_intervention]

            sum_row_firings = df_transfer[firing_name].sum()
            df_to_analise.loc[4] = ['Sum of Duration Times', sum_row_firings]

            mean_row_firings = df_transfer[firing_name].mean()
            df_to_analise.loc[5] = ['Average of Duration Time per Hour', mean_row_firings]

            max_row__index_firing = df_transfer[firing_name].idxmax()
            max_row_firing = df_transfer.loc[max_row__index_firing, firing_name].tolist()
            df_to_analise.loc[6] = ['Max of Duration Time per Hour', max_row_firing]

            min_row__index_firing = df_transfer[firing_name].idxmin() 
            min_row_firing = df_transfer.loc[min_row__index_firing, firing_name].tolist()
            df_to_analise.loc[7] = ['Min of Duration Time per Hour', min_row_firing]

        # Połączenie ramek danych
        df_data = pd.concat([df_transfer, df_duration], axis=1)

        if self.presentation_mode == 'chart':
            self.draw_chart(df_transfer, df_duration)
            self.analyse(df_to_analise, df_transfer, df_duration)
        elif self.presentation_mode == 'table':
            self.create_table(df_data)

        # Utworzenie buttona do exportu danych
        button = ttk.Button(self.export_frame, text="Export data", command=lambda: export_to_csv(df_data, f"Comparison Of Firings And Inteventions - {self.city_var.get()}"))
        button.grid(row=0, column=0, padx=15, pady=5, sticky="nsew")


    # Rysowanie wykresu
    def draw_chart(self, df_transfer, df_duration):
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

        if self.data_type_dropdown_var.get() == 'Firings and Interventions':
            if self.mode_dropdown_var.get() == "Amount" or self.mode_dropdown_var.get() == "Duration" or self.mode_dropdown_var.get() == "Average Duration" :
                if self.mode_dropdown_var.get() == "Amount":
                    y_label_intervention = "amountOfInterventions"
                    y_label_firing = "amountOfFirings"
                    y_label_name = 'Amount Of Incidents'
                if self.mode_dropdown_var.get() == "Duration":
                    y_label_intervention = "interventionsDuration[h]"
                    y_label_firing = "firingsDuration[h]"
                    y_label_name = 'Duration Of Incidents[h]'
                if self.mode_dropdown_var.get() == "Average Duration":
                    y_label_intervention = "averageInterventionDuration[min]"
                    y_label_firing = "averageFiringDuration[min]"
                    y_label_name = 'Average Duration Of Incident'
                
                # Rysowanie wykresu słupkowego
                bar_width = 0.35
                bar_positions = np.arange(len(df_duration.index)) + 1

                plt.bar(bar_positions - bar_width/2, df_duration[y_label_intervention], label="Intervention", width=bar_width)
                plt.bar(bar_positions + bar_width/2, df_duration[y_label_firing], label="Firing", width=bar_width)

                # Dodanie etykiet oraz tytułu wykresu
                plt.xlabel("Per Hour [h]")
                plt.ylabel(y_label_name)
                plt.title(f"{self.data_type_dropdown_var.get()} per Hour for {self.city_var.get()}")
            
            elif self.mode_dropdown_var.get() == "Average Transfer":
                y_label_intervention = "averageTransferToInterventionDuration[min]"
                y_label_firing = "averageTransferToFiringDuration[min]"
                y_label_name = 'Transfer Time[min]'

                # Rysowanie wykresu słupkowego
                bar_width = 0.35
                bar_positions = np.arange(len(df_transfer.index)) + 1

                plt.bar(bar_positions - bar_width/2, df_transfer[y_label_intervention], label="Intervention", width=bar_width)
                plt.bar(bar_positions + bar_width/2, df_transfer[y_label_firing], label="Firing", width=bar_width)

                # Dodanie etykiet oraz tytułu wykresu
                plt.xlabel("Hour[h]")
                plt.ylabel(y_label_name)
                plt.title(f"{self.data_type_dropdown_var.get()} per Hour for {self.city_var.get()}")

            elif self.mode_dropdown_var.get() == "Incidents vs Transfer Time":

                y_label_time = "averageTransferToIncidents[min]"
                y_label_incidents = "amountOfIncidents"
                y_label_name = 'Transfer Time[min] | Incidents'

                # Rysowanie wykresu słupkowego
                bar_width = 0.35
                bar_positions = np.arange(len(df_transfer.index)) + 1

                plt.bar(bar_positions - bar_width/2, df_transfer[y_label_time], label="Transfer Time", width=bar_width)
                plt.bar(bar_positions + bar_width/2, df_duration[y_label_incidents], label="Amount of Incidents", width=bar_width)

                # Dodanie etykiet oraz tytułu wykresu
                plt.xlabel("Hour[h]")
                plt.ylabel(y_label_name)
                plt.title(f"{self.data_type_dropdown_var.get()} per Hour for {self.city_var.get()}")
        
        elif self.data_type_dropdown_var.get() == 'Firings':
            if self.mode_dropdown_var.get() == "Amount" or self.mode_dropdown_var.get() == "Duration" or self.mode_dropdown_var.get() == "Average Duration" :
                if self.mode_dropdown_var.get() == "Amount":
                    y_label_firing = "amountOfFirings"
                    y_label_name = 'Amount Of Firings'
                if self.mode_dropdown_var.get() == "Duration":
                    y_label_firing = "firingsDuration[h]"
                    y_label_name = 'Duration Of Firings[h]'
                if self.mode_dropdown_var.get() == "Average Duration":
                    y_label_firing = "averageFiringDuration[min]"
                    y_label_name = 'Average Duration Of Firings'
                
                # Rysowanie wykresu słupkowego
                plt.bar(df_duration.index + 1, df_duration[y_label_firing], label="Firing")

                # Dodanie etykiet oraz tytułu wykresu
                plt.xlabel("Per Hour [h]")
                plt.ylabel(y_label_name)
                plt.title(f"{self.data_type_dropdown_var.get()} per Hour for {self.city_var.get()}")
            
            elif self.mode_dropdown_var.get() == "Average Transfer":
                y_label_firing = "averageTransferToFiringDuration[min]"
                y_label_name = 'Transfer To Firing Time[min]'

                # Rysowanie wykresu słupkowego
                plt.bar(df_transfer.index + 1, df_transfer[y_label_firing], label="Firing")

                # Dodanie etykiet oraz tytułu wykresu
                plt.xlabel("Hour[h]")
                plt.ylabel(y_label_name)
                plt.title(f"{self.data_type_dropdown_var.get()} per Hour for {self.city_var.get()}")

            elif self.mode_dropdown_var.get() == "Incidents vs Transfer Time":

                y_label_time = "averageTransferToFiringDuration[min]"
                y_label_incidents = "amountOfFirings"
                y_label_name = 'Transfer Time[min] | Firings'

                # Rysowanie wykresu słupkowego
                bar_width = 0.35
                bar_positions = np.arange(len(df_transfer.index)) + 1

                plt.bar(bar_positions - bar_width/2, df_transfer[y_label_time], label="Transfer Time", width=bar_width)
                plt.bar(bar_positions + bar_width/2, df_duration[y_label_incidents], label="Amount of Firings", width=bar_width)

                # Dodanie etykiet oraz tytułu wykresu
                plt.xlabel("Hour[h]")
                plt.ylabel(y_label_name)
                plt.title(f"{self.data_type_dropdown_var.get()} per Hour for {self.city_var.get()}")

        elif self.data_type_dropdown_var.get() == 'Interventions':
            if self.mode_dropdown_var.get() == "Amount" or self.mode_dropdown_var.get() == "Duration" or self.mode_dropdown_var.get() == "Average Duration" :
                if self.mode_dropdown_var.get() == "Amount":
                    y_label_intervention = "amountOfInterventions"
                    y_label_name = 'Amount Of Interventions'
                if self.mode_dropdown_var.get() == "Duration":
                    y_label_intervention = "interventionsDuration[h]"
                    y_label_name = 'Duration Of Interventions[h]'
                if self.mode_dropdown_var.get() == "Average Duration":
                    y_label_intervention = "averageInterventionDuration[min]"
                    y_label_name = 'Average Duration Of Interventions'
            
                # Rysowanie wykresu słupkowego
                plt.bar(df_duration.index + 1, df_duration[y_label_intervention], label="Intervention")

                # Dodanie etykiet oraz tytułu wykresu
                plt.xlabel("Per Hour [h]")
                plt.ylabel(y_label_name)
                plt.title(f"{self.data_type_dropdown_var.get()} per Hour for {self.city_var.get()}")
            
            elif self.mode_dropdown_var.get() == "Average Transfer":
                y_label_intervention = "averageTransferToInterventionDuration[min]"
                y_label_name = 'Transfer To Interventions Time[min]'

                # Rysowanie wykresu słupkowego
                plt.bar(df_transfer.index + 1, df_transfer[y_label_intervention], label="Intervention")

                # Dodanie etykiet oraz tytułu wykresu
                plt.xlabel("Hour[h]")
                plt.ylabel(y_label_name)
                plt.title(f"{self.data_type_dropdown_var.get()} per Hour for {self.city_var.get()}")

            elif self.mode_dropdown_var.get() == "Incidents vs Transfer Time":

                y_label_time = "averageTransferToInterventionDuration[min]"
                y_label_incidents = "amountOfInterventions"
                y_label_name = 'Transfer Time[min] | Interventions'

                # Rysowanie wykresu słupkowego
                bar_width = 0.35
                bar_positions = np.arange(len(df_transfer.index)) + 1

                plt.bar(bar_positions - bar_width/2, df_transfer[y_label_time], label="Transfer Time", width=bar_width)
                plt.bar(bar_positions + bar_width/2, df_duration[y_label_incidents], label="Amount of Interventions", width=bar_width)

                # Dodanie etykiet oraz tytułu wykresu
                plt.xlabel("Hour[h]")
                plt.ylabel(y_label_name)
                plt.title(f"{self.data_type_dropdown_var.get()} per Hour for {self.city_var.get()}")

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
        button = ttk.Button(self.export_frame, text="Export chart", command=lambda: export_plot_to_image(fig, f"Comparison Of Firings And Inteventions - {self.city_var.get()}"))
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
    def analyse(self, data, df_transfer, df_duration):
        # Wyczyszczenie frame
        for widget in self.frame_analise.winfo_children():
            widget.destroy()
        
        if self.mode_dropdown_var.get() == "Amount":
            unit = ''
            y_title = 'Amount'            
            if self.data_type_dropdown_var.get() == "Firings and Interventions":
                first_data = 'amountOfInterventions'
                second_data = 'amountOfFirings'
                title = 'Amount of Interventions and Firings'
            elif self.data_type_dropdown_var.get() == "Firings":
                first_data = 'amountOfFirings'
                title = 'Amount of Firings'
            elif self.data_type_dropdown_var.get() == "Interventions":
                first_data = 'amountOfInterventions'
                title = 'Amount of Interventions'
        elif self.mode_dropdown_var.get() == "Duration":
            unit = 'h'
            y_title = 'Sum Of Durations[h]'            
            if self.data_type_dropdown_var.get() == "Firings and Interventions":
                first_data = 'interventionsDuration[h]'
                second_data = 'firingsDuration[h]'
                title = 'Duration of Interventions and Firings'
            elif self.data_type_dropdown_var.get() == "Firings":
                first_data = 'firingsDuration[h]'
                title = 'Duration of Firing'
            elif self.data_type_dropdown_var.get() == "Interventions":
                first_data = 'interventionsDuration[h]'
                title = 'Duration of Interventions'
        elif self.mode_dropdown_var.get() == "Average Duration":
            unit = 'min'
            y_title = 'Average Duration[min]'            
            if self.data_type_dropdown_var.get() == "Firings and Interventions":
                first_data = 'averageInterventionDuration[min]'
                second_data = 'averageFiringDuration[min]'
                title = 'Average Duration of Interventions and Firings'
            elif self.data_type_dropdown_var.get() == "Firings":
                first_data = 'averageFiringDuration[min]'
                title = 'Average Duration of Firings'
            elif self.data_type_dropdown_var.get() == "Interventions":
                first_data = 'averageInterventionDuration[min]'
                title = 'Average Duration of Interventions'
        elif self.mode_dropdown_var.get() == "Average Transfer":
            unit = 'min'
            y_title = 'Average Transfer[min]'            
            if self.data_type_dropdown_var.get() == "Firings and Interventions":
                first_data = 'averageTransferToInterventionDuration[min]'
                second_data = 'averageTransferToFiringDuration[min]'
                title = 'Average Transfer to Interventions and Firings'
            elif self.data_type_dropdown_var.get() == "Firings":
                first_data = 'averageTransferToFiringDuration[min]'
                title = 'Average Transfer to Firing'
            elif self.data_type_dropdown_var.get() == "Interventions":
                first_data = 'averageTransferToInterventionDuration[min]'
                title = 'Average Transfer to Intervention'
        elif self.mode_dropdown_var.get() == "Incidents vs Transfer Time":
            unit = ''
            y_title = 'Incidents And Transfer Time'            
            if self.data_type_dropdown_var.get() == "Firings and Interventions":
                first_data = 'averageTransferToIncidents[min]'
                second_data = 'amountOfIncidents'
                title = "All Incidents vs Transfer Time"
            elif self.data_type_dropdown_var.get() == "Firings":
                first_data = 'averageTransferToFiringDuration[min]'
                second_data = 'amountOfFirings'
                title = "Firings vs Transfer Time"
            elif self.data_type_dropdown_var.get() == "Interventions":
                first_data = 'averageTransferToInterventionDuration[min]'
                second_data = 'amountOfInterventions'
                title = "Interventions vs Transfer Time"
            
        # Utwórzenie drzewa do wyświetlania danych
        tree = ttk.Treeview(self.frame_analise, columns=["Option", "Value"], show="headings")

        # Dodanie kolumny
        tree.heading("Option", text="Operation")
        tree.heading("Value", text="Value")

        tree.column("#1", anchor="center")
        tree.column("#2", anchor="center")

        for index, row in data.iterrows():
            tree.insert("", "end", values=(row['Operation'], f"{row['Value']:.2f} {unit}"))

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
        
        if self.mode_dropdown_var.get() == "Amount" or self.mode_dropdown_var.get() == "Duration" or self.mode_dropdown_var.get() == "Average Duration":
            if self.data_type_dropdown_var.get() == "Firings and Interventions":
                plt.plot(df_duration.index + 1, df_duration[first_data], label="Intervention")
                plt.plot(df_duration.index + 1, df_duration[second_data], label="Firing")
            else:
                plt.plot(df_duration.index + 1, df_duration[first_data], label="Incident")
        elif self.mode_dropdown_var.get() == "Average Transfer":
            if self.data_type_dropdown_var.get() == "Firings and Interventions":
                plt.plot(df_duration.index + 1, df_transfer[first_data], label="Intervention")
                plt.plot(df_duration.index + 1, df_transfer[second_data], label="Firing")
            else:
                plt.plot(df_duration.index + 1, df_transfer[first_data], label="Incident")
        elif self.mode_dropdown_var.get() == "Incidents vs Transfer Time":
            plt.plot(df_duration.index + 1, df_transfer[first_data], label="Transfer Time")
            plt.plot(df_duration.index + 1, df_duration[second_data], label="Incidents")

        # Ustawienia osi i etykiet
        ax.set_xlabel('Hour[h]')
        ax.set_ylabel(y_title)
        ax.set_title(title)
        ax.legend()

        # Ustawienie przyrostów na osiach
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(1))

        fig.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.90)

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