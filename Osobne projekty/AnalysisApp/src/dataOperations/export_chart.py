from tkinter import filedialog
import os

def export_plot_to_image(figure, default_file_name="plot_image.png"):
    exports_folder = "exports"

    if not os.path.exists(exports_folder):
        os.makedirs(exports_folder)

    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        initialfile=default_file_name,
        initialdir=exports_folder
    )

    if file_path:
        figure.savefig(file_path, bbox_inches='tight')
        print(f"Wykres został pomyślnie zapisany do pliku: {file_path}")
    else:
        print("Operacja zapisu anulowana.")

