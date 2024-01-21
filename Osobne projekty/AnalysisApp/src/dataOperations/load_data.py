import glob
import pandas as pd

def load_data(selected_cities, chart_title):
    # Utwórz pusty DataFrame
    combined_df = pd.DataFrame()

    for city in selected_cities:
        # Utwórz nazwę pliku
        file_pattern = f"results/{city}/*--{chart_title}.csv"
        
        # Uzyskaj listę plików pasujących do wzorca
        matching_files = glob.glob(file_pattern)

        # Sprawdź, czy znaleziono pasujące pliki
        if not matching_files:
            print(f"No matching file found in directory: {file_pattern}")
            continue

        # Wybierz pierwszy pasujący plik (w razie gdyby było więcej)
        selected_file = matching_files[0]
        
        # Odczytaj dane z wybranego pliku CSV
        try:
            df = pd.read_csv(selected_file, encoding='Windows-1250')
            print("Loaded data from: " + selected_file)
            
            # Dodaj kolumnę 'City' z nazwą miasta
            df['City'] = city
            
            # Dołącz DataFrame do głównego DataFrame
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except UnicodeDecodeError:
            print("Unsuccessful attempt to read CSV file.")

    return combined_df