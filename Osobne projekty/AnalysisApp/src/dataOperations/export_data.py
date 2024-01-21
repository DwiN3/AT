from tkinter import filedialog
import os

def export_to_csv(data_frame, default_file_path="exported_data.csv"):
    exports_folder = "exports"

    if not os.path.exists(exports_folder):
        os.makedirs(exports_folder)

    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        initialfile=default_file_path,
        initialdir=exports_folder
    )

    if file_path:
        data_frame.to_csv(file_path, index=False)
        print(f"Dane zostały pomyślnie zapisane do pliku CSV: {file_path}")
    else:
        print("Operacja zapisu anulowana.")