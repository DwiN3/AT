import pandas as pd
from matplotlib import pyplot as plt

# Ścieżka do pliku CSV
file_path = r"/results/Berlin/12-11-2023_20-00-38--Changing State Details.csv"
df = pd.read_csv(file_path, encoding='latin1')

patrol_ids = df['patrolID'].unique()
patrol_counts = df['patrolID'].value_counts()
patrol_ids_more = patrol_counts[patrol_counts > 1].index.tolist()
district_name = df['districtName'].unique()
states = ["PATROLLING","INTERVENTION","CALCULATING_PATH","TRANSFER_TO_INTERVENTION","TRANSFER_TO_FIRING","FIRING","NEUTRALIZED","RETURNING_TO_HQ",]

# Settings
display_mode = "Time"            # Time / Procent / Value

patrol = patrol_ids[0]                   # patrol_ids[0]
districtName = "All"             # All / district_name[0]
districtSafetyLevel = "All"      # All / Safe / NotSafe / RatherSafe
currentState = "All"             # All / states[0]
isNight = "All"                  # All / day / night

# Filtracja danych na podstawie ustawień
if patrol != "All":
    df = df[df['patrolID'] == patrol]

if districtName != "All":
    df = df[df['districtName'] == districtName]

if districtSafetyLevel != "All":
    df = df[df['districtSafetyLevel'] == districtSafetyLevel]

if currentState != "All":
    df = df[df['currentPatrolState'] == currentState]

if isNight != "All":
    if isNight == "day":
        df = df[df['isNight'] == 0]
    else:
        df = df[df['isNight'] == 1]

if display_mode == "Time":
    plt.figure(figsize=(10, 6))
    for patrol_id, group in df.groupby('patrolID'):
        plt.scatter(group['simulationTime[s]'], [patrol_id] * len(group), label=patrol_id)
    plt.title('Ilość patroli w zależności od czasu')
    plt.xlabel('Czas symulacji [s]')
    plt.ylabel('Ilość patrolu')
    plt.legend(title='Patrol ID', bbox_to_anchor=(1.05, 1), loc='upper left')

if display_mode == "Procent":
    filtered_count = len(df)  # Całkowita liczba wierszy po zastosowaniu filtrowania
    patrol_count = len(df[df['patrolID'] == patrol])  # Liczba wierszy dla wybranego patrolID

    if filtered_count == 0:
        percentage_value = 0  # Uniknięcie dzielenia przez zero
    else:
        percentage_value = (patrol_count / filtered_count) * 100

    # Tworzenie wykresu słupkowego
    fig, ax = plt.subplots()
    ax.bar(['Filtered Data'], [percentage_value], color='blue')
    ax.set_ylabel('Percentage')
    ax.set_title(f'Percentage of {patrol} in Filtered Data')

    # Wyświetlenie wartości procentowej na słupku
    for i, value in enumerate([percentage_value]):
        ax.text(i, value + 1, f'{value:.2f}%', ha='center', va='bottom')

plt.show()