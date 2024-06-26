from datetime import datetime, timedelta
import json
import customtkinter
from CTkListbox import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkcalendar import DateEntry

from backend import prediction
from backend.DataManagement import load_countries, load_data
from Plotting import plot_country_chart
from WindowUtils import calculate_window_size, center_window

current_chart_index = 0
charts = []
data_for_charts = None
manual_entry_result = None
search_query = ""


def search_country_in_file(country_name):
    try:
        with open("data/Countries.json", "r", encoding="utf-8") as f:
            countries_data = json.load(f)
            return country_name.lower() in [country.lower() for country in countries_data["countries"]]
    except FileNotFoundError:
        return False


def main():
    global current_chart_index, manual_entry_result, search_query

    root = customtkinter.CTk()
    root.title("Corona Rush")
    root.iconbitmap("icon.ico")

    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("blue")

    def show_waiting_message():
        for widget in right_frame.winfo_children():
            widget.destroy()
        waiting_label = customtkinter.CTkLabel(right_frame, text="Waiting for data...", font=("Arial", 20))
        waiting_label.place(relx=0.5, rely=0.5, anchor="center")

    def show_current_chart():
        for widget in right_frame.winfo_children():
            widget.destroy()

        if charts:
            fig = charts[current_chart_index]
            canvas = FigureCanvasTkAgg(fig, master=right_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
        else:
            show_waiting_message()

    def on_prev():
        global current_chart_index
        if current_chart_index > 0:
            current_chart_index -= 1
        show_current_chart()

    def on_next():
        global current_chart_index
        if current_chart_index < len(charts) - 1:
            current_chart_index += 1
        show_current_chart()

    def on_submit_dates(dates):
        try:
            date1 = datetime.strptime(dates[0], "%d-%m-%Y")
            date2 = datetime.strptime(dates[1], "%d-%m-%Y")
            date3 = datetime.strptime(dates[2], "%d-%m-%Y")

            if date2 <= date1 or (date2 - date1).days < 89:
                log_message("The second date must be at least 3 months later than the first.")
                return
            if date3 <= date1 or date3 <= date2:
                log_message("The third date must be later than the second and first.")
                return
            global current_chart_index, charts

            if manual_entry_result:
                selected_country = search_query
            else:
                selected_country = listbox.get(listbox.curselection())

            data = load_data(selected_country)

            if not data:
                log_message(f"No data found for {selected_country}")
                return

            filtered_data = filter_data_by_dates(data, date1, date2)
            prediction_charts = prediction.make_prediction(filtered_data, date3.strftime("%d-%m-%Y"))
            country_chart_total = plot_country_chart(filtered_data, selected_country, 'total')
            country_chart_new = plot_country_chart(filtered_data, selected_country, 'new')
            charts = [country_chart_total, country_chart_new] + prediction_charts
            current_chart_index = 0
            show_current_chart()
        except ValueError as e:
            log_message(f"Error parsing dates: {e}")
        except KeyError as e:
            log_message(f"Key error: {e}")
            print(f"Key error: {e}")
            print(f"Data for {selected_country}: {data}")
        except Exception as e:
            log_message(f"Error: {e}")

    def filter_data_by_dates(data, start_date, end_date):
        """Filtruje dane na podstawie podanego zakresu dat."""
        filtered_cases = {date: details for date, details in data[0]['cases'].items() if
                          start_date <= datetime.strptime(date, '%Y-%m-%d') <= end_date}
        return [{'country': data[0]['country'], 'region': data[0]['region'], 'cases': filtered_cases}]

    def calculate_future_date():
        today = datetime.now()
        future_date = today + timedelta(days=365)
        return future_date

    def reset_dates():
        # Resetujemy daty do domyślnych wartości
        date_entry1.set_date(datetime(2019, 1, 1))
        date_entry2.set_date(datetime(2022, 12, 31))
        date_entry3.set_date(calculate_future_date())

    def log_message(message):
        alert_label.configure(text=message, text_color="white", font=("Arial", 20))
        print(message)
        # Change color back after 3 seconds
        alert_label.after(5000, lambda: alert_label.configure(text_color="#242424"))

    def open_manual_entry_window():
        manual_entry_window = customtkinter.CTkToplevel(root)
        manual_entry_window.title("Enter country manually")

        entry_label = customtkinter.CTkLabel(manual_entry_window, text="Enter country:")
        entry_label.pack(pady=10)

        country_entry = customtkinter.CTkEntry(manual_entry_window, width=250)
        country_entry.pack(pady=10)

        def manual_entry_submit():
            global manual_entry_result
            global search_query

            search_query = country_entry.get().strip().capitalize()
            if search_country_in_file(search_query):
                log_message(f"Selected country: {search_query}")
                manual_entry_result = True
                manual_entry_window.destroy()
            else:
                log_message("Incorrect country given!")

        button_frame = customtkinter.CTkFrame(manual_entry_window)
        button_frame.pack(pady=10)

        cancel_button = customtkinter.CTkButton(button_frame, text="Cancel", command=manual_entry_window.destroy)
        cancel_button.pack(side="left", padx=10)

        submit_button = customtkinter.CTkButton(button_frame, text="Accept", command=manual_entry_submit)
        submit_button.pack(side="right", padx=10)

        # Set the position of the manual entry window
        manual_entry_button_pos = manual_entry_button.winfo_rootx(), manual_entry_button.winfo_rooty()
        manual_entry_window.geometry(f"+{manual_entry_button_pos[0]}+{manual_entry_button_pos[1] - 100}")

        manual_entry_window.transient(root)
        manual_entry_window.grab_set()
        manual_entry_window.focus_set()
        manual_entry_window.attributes('-topmost', True)

    def sum_new_cases(data, start_date, end_date):
        total_new_cases = 0
        for date, details in data[0]['cases'].items():
            current_date = datetime.strptime(date, '%Y-%m-%d')
            if start_date <= current_date <= end_date:
                total_new_cases += details.get('new', 0)
        return total_new_cases

    def open_details_window():
        details_window = customtkinter.CTkToplevel(root)
        details_window.title("Details")

        # Ustawienie większego rozmiaru okna
        details_window.geometry("400x400")

        selected_country = search_query if manual_entry_result else listbox.get(listbox.curselection())
        date1_str = date_entry1.get()
        date2_str = date_entry2.get()
        date3_str = date_entry3.get()

        # Konwersja dat do obiektów datetime
        date1 = datetime.strptime(date1_str, "%d-%m-%Y")
        date2 = datetime.strptime(date2_str, "%d-%m-%Y")
        date3 = datetime.strptime(date3_str, "%d-%m-%Y")

        # Obliczanie różnic w dniach
        diff_from_to = (date2 - date1).days
        diff_to_prediction = (date3 - date2).days
        diff_from_prediction = (date3 - date1).days

        # Ładowanie danych dla wybranego kraju
        data = load_data(selected_country)

        # Sumowanie nowych przypadków w określonych przedziałach dat
        new_cases_from_to = sum_new_cases(data, date1, date2)
        new_cases_to_prediction = sum_new_cases(data, date2, date3)

        # Wyświetlanie wybranych informacji
        country_label = customtkinter.CTkLabel(details_window, text=f"Selected Country: {selected_country}")
        country_label.pack(pady=10)

        date1_label = customtkinter.CTkLabel(details_window, text=f"From: {date1_str}")
        date1_label.pack(pady=10)

        date2_label = customtkinter.CTkLabel(details_window, text=f"To: {date2_str}")
        date2_label.pack(pady=10)

        date3_label = customtkinter.CTkLabel(details_window, text=f"Prediction To: {date3_str}")
        date3_label.pack(pady=10)

        # Wyświetlanie różnic w dniach
        diff_from_to_label = customtkinter.CTkLabel(details_window, text=f"Difference (from - to): {diff_from_to} days")
        diff_from_to_label.pack(pady=10)

        diff_to_prediction_label = customtkinter.CTkLabel(details_window,
                                                          text=f"Difference (to - prediction to): {diff_to_prediction} days")
        diff_to_prediction_label.pack(pady=10)

        diff_from_prediction_label = customtkinter.CTkLabel(details_window,
                                                            text=f"Difference (from - prediction to): {diff_from_prediction} days")
        diff_from_prediction_label.pack(pady=10)

        # Wyświetlanie sum nowych przypadków
        new_cases_from_to_label = customtkinter.CTkLabel(details_window,
                                                         text=f"New Cases (from - to): {new_cases_from_to}")
        new_cases_from_to_label.pack(pady=10)

        new_cases_to_prediction_label = customtkinter.CTkLabel(details_window,
                                                               text=f"New Cases (to - prediction to): {new_cases_to_prediction}")
        new_cases_to_prediction_label.pack(pady=10)

        # Wyświetlanie okna zawsze na przodzie
        details_window.attributes('-topmost', True)

    def on_listbox_select(event):
        global manual_entry_result, search_query
        manual_entry_result = False
        search_query = listbox.get(listbox.curselection())
        log_message(f"Selected country: {search_query}")

    window_width, window_height = calculate_window_size(root.winfo_screenwidth(), root.winfo_screenheight())
    root.geometry(f"{window_width}x{window_height}")
    center_window(root, window_width, window_height)

    right_frame = customtkinter.CTkFrame(root, width=800, height=window_height - 100, border_width=5,
                                         border_color="#F9AA33")
    right_frame.place(relx=0.62, rely=0.47, anchor="center")

    show_waiting_message()

    left_frame = customtkinter.CTkFrame(root, width=300, height=window_height)
    left_frame.place(relx=0.145, rely=0.37, anchor="center")

    listbox = CTkListbox(left_frame, width=250, height=150)
    listbox.pack(pady=10)
    listbox.bind("<<ListboxSelect>>", on_listbox_select)

    manual_entry_button = customtkinter.CTkButton(left_frame, text="Enter manually", command=open_manual_entry_window)
    manual_entry_button.pack(pady=5)

    countries = load_countries()
    for i, country in enumerate(countries):
        listbox.insert(i, country)

    date_label = customtkinter.CTkLabel(left_frame, text="Select three dates (DD-MM-YYYY):")
    date_label.pack(pady=5)

    date_frame = customtkinter.CTkFrame(left_frame)
    date_frame.pack(pady=5)

    # Pierwsza data
    date_entry1_date = datetime(2019, 1, 1)
    date_label1 = customtkinter.CTkLabel(date_frame, text="from:")
    date_label1.grid(row=0, column=0, padx=5)
    date_entry1 = DateEntry(date_frame, width=18, background="black", disabledbackground="black", bordercolor="white",
                            headersbackground="#242424", normalbackground="black", foreground='white',
                            normalforeground='white', headersforeground='white', borderwidth=2,
                            date_pattern='dd-mm-yyyy')
    date_entry1.set_date(date_entry1_date)
    date_entry1.grid(row=0, column=1, padx=5)

    # Druga data
    date_entry2_date = datetime(2022, 12, 31)
    date_label2 = customtkinter.CTkLabel(date_frame, text="to:")
    date_label2.grid(row=1, column=0, padx=5)
    date_entry2 = DateEntry(date_frame, width=18, background="black", disabledbackground="black", bordercolor="white",
                            headersbackground="#242424", normalbackground="black", foreground='white',
                            normalforeground='white', headersforeground='white', borderwidth=2,
                            date_pattern='dd-mm-yyyy')
    date_entry2.set_date(date_entry2_date)

    # Zabezpieczenie
    date_entry2.max_date = datetime.now().date()
    date_entry2.grid(row=1, column=1, padx=5)

    # Trzecia data
    date_entry3_date = calculate_future_date()
    date_label3 = customtkinter.CTkLabel(date_frame, text="prediction to:")
    date_label3.grid(row=2, column=0, padx=5)
    date_entry3 = DateEntry(date_frame, width=18, background="black", disabledbackground="black", bordercolor="white",
                            headersbackground="#242424", normalbackground="black", foreground='white',
                            normalforeground='white', headersforeground='white', borderwidth=2,
                            date_pattern='dd-mm-yyyy')
    date_entry3.set_date(date_entry3_date)
    date_entry3.grid(row=2, column=1, padx=5)

    start_reset_frame = customtkinter.CTkFrame(left_frame)
    start_reset_frame.pack(side="top", pady=10)

    submit_button = customtkinter.CTkButton(start_reset_frame, text="Start", command=lambda: on_submit_dates(
        [date_entry1.get(), date_entry2.get(), date_entry3.get()]))
    submit_button.pack(side="right", padx=10)

    reset_button = customtkinter.CTkButton(start_reset_frame, text="Reset data range", command=lambda: reset_dates())
    reset_button.pack(side="left", padx=10)

    # Ramka dla alertów
    alert_frame = customtkinter.CTkFrame(root, height=50, fg_color="#242424")
    alert_frame.pack(fill='x', side='bottom')

    # Etykieta dla wyświetlania alertów
    alert_label = customtkinter.CTkLabel(alert_frame, text="", font=("Arial", 20))
    alert_label.pack(pady=15)

    navigation_frame = customtkinter.CTkFrame(left_frame)
    navigation_frame.pack(pady=20)

    prev_button = customtkinter.CTkButton(navigation_frame, text="Previous", command=on_prev)
    prev_button.pack(side="left", padx=10)

    next_button = customtkinter.CTkButton(navigation_frame, text="Next", command=on_next)
    next_button.pack(side="right", padx=10)

    details_button = customtkinter.CTkButton(left_frame, text="Show Details", command=open_details_window, fg_color="#F9AA33", hover_color="#956720", text_color="#242424")
    details_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()