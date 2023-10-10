import math
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import levy_flight as lf
import cuckoo as ck

population_size = ""
max_iterations = ""
probability = ""
lower_bound = ""
upper_bound = ""

population_size = ""
max_iterations = ""
probability = ""
lower_bound = ""
upper_bound = ""

def run_algorithm():
    if not calculate_button["state"] == "disabled":
        plot_clear()
        # Pobieranie wartości z pól ustawień
        population_size_input = table_frame.grid_slaves(row=0, column=2)[0].get()
        max_iterations_input = table_frame.grid_slaves(row=1, column=2)[0].get()
        probability_input = table_frame.grid_slaves(row=2, column=2)[0].get()
        lower_bound_input = table_frame.grid_slaves(row=3, column=2)[0].get()
        upper_bound_input = table_frame.grid_slaves(row=4, column=2)[0].get()

        if population_size_input and max_iterations_input and probability_input and lower_bound_input and upper_bound_input:
            err_label.config(text="")
            
            # Konwersja wartości na odpowiednie typy
            population_size = int(population_size_input)
            max_iterations = int(max_iterations_input)
            probability = float(probability_input)
            lower_bound = int(lower_bound_input)
            upper_bound = int(upper_bound_input)
            name_function = choice_box.get()
            print(name_function)

            # Wywołanie algorytmu kukułczego
            best_solution = ck.cuckoo_search_algorithm(population_size, max_iterations, lower_bound, upper_bound, probability, name_function)
            top_best_solution = best_solution[:5]
            plot(best_solution)

            # Ustawienie wartości w etykietach wyników
            for i in range(rows):
                if i < len(best_solution):
                    result_labels[i].config(text=f"{i + 1}.  {best_solution[i]}", anchor=tk.W)
                else:
                    result_labels[i].config(text=f"Wynik {i + 1}:", anchor=tk.W)

            # Podmiana wartości "Wynik 1" na wartość top_best_solution[0]
            if top_best_solution:
                best_result_value_label.config(text=top_best_solution[0])
        else:
            err_label.config(text="Uzupełnij wszystkie pola")


def plot(solution):
    x = [0]
    y = [0]

    for i in range(len(solution)):
        dx = solution[i] * math.sin(i)
        dy = solution[i] * math.cos(i)

        x.append(x[-1] + dx)
        y.append(y[-1] + dy)

    # Tworzenie pierwszego wykresu 'The Best Solution'
    fig1 = Figure(figsize=(5, 4), dpi=75)
    plot1 = fig1.add_subplot(111)
    plot1.plot(x, y, linestyle='-', marker='', color='red')
    plot1.set_xlabel('X')
    plot1.set_ylabel('Y')
    plot1.set_title("Wykres 'The Best Solution'")

    canvas1 = FigureCanvasTkAgg(fig1, master=chart1_frame)
    canvas1.draw()
    canvas1.get_tk_widget().pack()

    # Tworzenie drugiego wykresu 'Levy Flight'
    step_length = 1
    lambda_value = 1.5
    num_steps = 1000
    x_l, y_l = lf.levy_flight_plot(step_length, lambda_value, num_steps)

    fig2 = Figure(figsize=(5, 4), dpi=75)
    plot2 = fig2.add_subplot(111)
    plot2.plot(x_l, y_l, linestyle='-', marker='', color='blue')
    plot2.set_xlabel('X')
    plot2.set_ylabel('Y')
    plot2.set_title("Wykres lotów Levy'ego")

    canvas2 = FigureCanvasTkAgg(fig2, master=chart2_frame)
    canvas2.draw()
    canvas2.get_tk_widget().pack()


def reset_results():
    # Czyszczenie etykiet wyników
    for label in result_labels:
        label.config(text="-")

    population_size = ""
    max_iterations = ""
    probability = ""
    lower_bound = ""
    upper_bound = ""
    best_result_value_label.config(text="-")
    table_frame.grid_slaves(row=0, column=2)[0].delete(0, 'end')
    table_frame.grid_slaves(row=0, column=2)[0].insert(0, "")
    table_frame.grid_slaves(row=1, column=2)[0].delete(0, 'end')
    table_frame.grid_slaves(row=1, column=2)[0].insert(0, "")
    table_frame.grid_slaves(row=2, column=2)[0].delete(0, 'end')
    table_frame.grid_slaves(row=2, column=2)[0].insert(0, "")
    table_frame.grid_slaves(row=3, column=2)[0].delete(0, 'end')
    table_frame.grid_slaves(row=3, column=2)[0].insert(0, "")
    table_frame.grid_slaves(row=4, column=2)[0].delete(0, 'end')
    table_frame.grid_slaves(row=4, column=2)[0].insert(0, "")
    # Czyszczenie etykiety najlepszego wyniku
    best_result_value_label.config(text="-")
    choice_box.selection_clear()
    choice_box.set("")
    # Czyszczenie wykresów
    plot_clear()

def set_default():
    reset_results()
    table_frame.grid_slaves(row=0, column=2)[0].delete(0, 'end')
    table_frame.grid_slaves(row=0, column=2)[0].insert(0, "100")
    table_frame.grid_slaves(row=1, column=2)[0].delete(0, 'end')
    table_frame.grid_slaves(row=1, column=2)[0].insert(0, "50")
    table_frame.grid_slaves(row=2, column=2)[0].delete(0, 'end')
    table_frame.grid_slaves(row=2, column=2)[0].insert(0, "0.25")
    table_frame.grid_slaves(row=3, column=2)[0].delete(0, 'end')
    table_frame.grid_slaves(row=3, column=2)[0].insert(0, "-10")
    table_frame.grid_slaves(row=4, column=2)[0].delete(0, 'end')
    table_frame.grid_slaves(row=4, column=2)[0].insert(0, "10")

def plot_clear():
    for widget in chart1_frame.winfo_children():
        widget.destroy()
    for widget in chart2_frame.winfo_children():
        widget.destroy()

root = tk.Tk()

# Ustawienia głównego okna
root.title("Algorytm kukułczy")

main_frame = tk.Frame(root, padx=20, pady=20)
main_frame.pack()

# Górna sekcja
upper_frame = tk.Frame(main_frame, pady=10)
upper_frame.pack()

# Wczytanie obrazu
image = Image.open("imgs/cuckoo_icon.png")

# Skalowanie obrazu do pożądanego rozmiaru
width, height = 60, 60
image = image.resize((width, height), Image.ANTIALIAS)

# Konwersja obrazu do formatu obsługiwanego przez tkinter
photo = ImageTk.PhotoImage(image)

# Utworzenie etykiety z obrazem
logo_label = tk.Label(upper_frame, image=photo)
logo_label.image = photo
logo_label.pack(side=tk.LEFT)

title_label = tk.Label(upper_frame, text="Algorytm kukułczy", font=("Helvetica", 19, "bold"))
title_label.pack(side=tk.LEFT)

# Kontener przechowujący Ustawienia, Wyniki i Wykresy
main_frame = tk.Frame(main_frame)
main_frame.pack(pady=10)

# Kontener z ustawieniami
settings_frame = ttk.Frame(main_frame, width=400, height=400, borderwidth=1, relief=tk.SOLID)
settings_frame.pack(side=tk.LEFT, padx=(0, 10), fill=tk.Y, pady=(20, 20))

# Wewnętrzny kontener z większym padding w kontenerze ustawień
inner_settings_frame = ttk.Frame(settings_frame, padding=20)
inner_settings_frame.pack(fill=tk.BOTH, expand=True)

settings_label = ttk.Label(inner_settings_frame, text="Ustawienia", font=("Helvetica", 14))
settings_label.pack(pady=(0, 10))

separator = ttk.Separator(inner_settings_frame, orient=tk.HORIZONTAL)
separator.pack(fill=tk.X, padx=10, pady=10)

table_frame = ttk.Frame(inner_settings_frame)
table_frame.pack()

# Tabela w kontenerze z ustawieniami
rows = 5
columns = 3

for i in range(rows):
    for j in range(columns):
        if j == 2:
            text_entry = ttk.Entry(table_frame)
            text_entry.grid(row=i, column=j, padx=10, pady=5)
        else:
            if i == 0 and j == 0:
                label = ttk.Label(table_frame, text="Rozmiar populacji", anchor='w', justify='left')
            elif i == 0 and j == 1:
                label = ttk.Label(table_frame, text="(N)", justify='center')
            elif i == 1 and j == 0:
                label = ttk.Label(table_frame, text="Ilość iteracji", anchor='w', justify='left')
            elif i == 1 and j == 1:
                label = ttk.Label(table_frame, text="(iter)", justify='center')
            elif i == 2 and j == 0:
                label = ttk.Label(table_frame, text="Prawdopodobieństwo", anchor='w', justify='left')
            elif i == 2 and j == 1:
                label = ttk.Label(table_frame, text="(prob)", justify='center')
            elif i == 3 and j == 0:
                label = ttk.Label(table_frame, text="Dolna granica", anchor='w', justify='left')
            elif i == 3 and j == 1:
                label = ttk.Label(table_frame, text="(Lb)", justify='center')
            elif i == 4 and j == 0:
                label = ttk.Label(table_frame, text="Górna granica", anchor='w', justify='left')
            elif i == 4 and j == 1:
                label = ttk.Label(table_frame, text="(Ub)", justify='center')
            if j == 0:
                label.grid(row=i, column=j, padx=10, pady=5, sticky='w')
            else:
                label.grid(row=i, column=j, padx=10, pady=5)

# Napis i rozwijany choice box
text_label = ttk.Label(inner_settings_frame, text="Wybierz funkcje celu:", anchor='center')
text_label.pack(pady=10)

options = ["rosenbrock", "rastrigin", "schwefel", "M/M/m/FIFO/m+N"]
choice_box = ttk.Combobox(inner_settings_frame, values=options, width=40, justify='center')
choice_box.pack()

# Przyciski Reset i Oblicz
buttons_frame = ttk.Frame(inner_settings_frame)
buttons_frame.pack(pady=20)

reset_button = ttk.Button(buttons_frame, text="Reset", width=19, command=reset_results)
reset_button.pack(side=tk.LEFT, padx=10)

# Przycisk Oblicz
calculate_button = ttk.Button(buttons_frame, text="Oblicz", width=19, command=run_algorithm)
calculate_button.pack(side=tk.LEFT, padx=10)

reset_button = ttk.Button(buttons_frame, text="Domyślne dane", width=19, command=set_default)
reset_button.pack(side=tk.LEFT, padx=10)

err_label = ttk.Label(inner_settings_frame, text="", font=("Helvetica", 10), justify='center')
err_label.pack(pady=(0, 10))



# Kontener z wynikami
results_frame = ttk.Frame(main_frame, width=400, height=400, borderwidth=1, relief=tk.SOLID)
results_frame.pack(side=tk.LEFT, padx=(0, 0), fill=tk.Y, pady=(20, 20))

# Wewnętrzny kontener z większym padding w kontenerze wyników
inner_results_frame = ttk.Frame(results_frame, padding=20)
inner_results_frame.pack(fill=tk.BOTH, expand=True)

results_label = ttk.Label(inner_results_frame, text="Wyniki", font=("Helvetica", 14) )
results_label.pack(pady=(0, 10))

separator2 = ttk.Separator(inner_results_frame, orient=tk.HORIZONTAL)
separator2.pack(fill=tk.X, padx=10, pady=10)

# Dodanie napisu "Optymalne Wyniki:"
optimal_results_label = ttk.Label(inner_results_frame, text="Optymalne Wyniki:", font=("Helvetica", 10), justify='left')
optimal_results_label.pack(pady=(0, 10))

# Tabela z optymalnymi wynikami
results_table_frame = ttk.Frame(inner_results_frame, borderwidth=1, relief=tk.SOLID)
results_table_frame.pack(padx=10, pady=0, fill=tk.BOTH, expand=True)

rows = 10
result_labels = []

for i in range(rows):
    label = ttk.Label(results_table_frame, text="-")
    label.pack(padx=10, pady=5, ipady=3)
    result_labels.append(label)

# Napis "Najlepszy wynik"
best_result_label = ttk.Label(inner_results_frame, text="Najlepszy wynik:", font=("Helvetica", 10))
best_result_label.pack(pady=10)

# Napis "123" pogrubiony pod etykietą "Najlepszy wynik"
best_result_value_label = ttk.Label(inner_results_frame, text="-", font=("Helvetica", 9, "bold"))
best_result_value_label.pack()

# Kontener z wykresami
charts_frame = ttk.Frame(main_frame, width=400, height=400, borderwidth=1, relief=tk.SOLID)
charts_frame.pack(side=tk.LEFT, padx=(10, 0), fill=tk.BOTH, pady=(20, 20))

# Wewnętrzny kontener z większym padding w kontenerze wykresów
inner_charts_frame = ttk.Frame(charts_frame, padding=20)
inner_charts_frame.pack(fill=tk.BOTH, expand=True)

charts_label = ttk.Label(inner_charts_frame, text="Wykresy", font=("Helvetica", 14))
charts_label.pack(pady=(0, 10))

charts_separator = ttk.Separator(inner_charts_frame, orient=tk.HORIZONTAL)
charts_separator.pack(fill=tk.X, padx=10, pady=10)

# Kontener dla pierwszego wykresu
chart1_container = ttk.Frame(inner_charts_frame, relief=tk.SOLID)
chart1_container.pack(side=tk.TOP, pady=(20, 10))

# Kontener dla drugiego wykresu
chart2_container = ttk.Frame(inner_charts_frame, relief=tk.SOLID)
chart2_container.pack(side=tk.TOP, pady=(0, 20))

# Kontener na pierwszy wykres
chart1_frame = ttk.Frame(chart1_container, width=350, height=350)
chart1_frame.pack()

chart1_label = ttk.Label(chart1_frame)
chart1_label.pack()

# Kontener na drugi wykres
chart2_frame = ttk.Frame(chart2_container, width=350, height=350)
chart2_frame.pack()

chart2_label = ttk.Label(chart2_frame)
chart2_label.pack()

root.mainloop()