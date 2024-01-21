import pandas as pd
import matplotlib.pyplot as plt

# Ustawienia
date = "firing"              # all, firing, intervention
mode = "averageTransfer"              # amount, duration, averageDuration, averageTransfer

# Wczytaj dane z pliku CSV
file_name_1 = r"E:\Projekty\Inzynieria_oprogramowania\AnalysisApp\results\Berlin\12-11-2023_20-00-38--Average Duration Of Incidents Per Hour.csv"
file_name_2 = r"E:\Projekty\Inzynieria_oprogramowania\AnalysisApp\results\Berlin\12-11-2023_20-00-38--Average Duration Patrols Heading Towards Incidents Per Hour.csv"

# Wczytaj dane do obiektu DataFrame
data = pd.read_csv(file_name_1 if mode != "averageTransfer" else file_name_2)

if mode != "averageTransfer":
    data['firingsDuration[min]'] /= 60
    data['interventionsDuration[min]'] /= 60
    data['averageFiringDuration[min]'] /= 60
    data['averageInterventionDuration[min]'] /= 60
else:
    data['averageTransferToInterventionDuration[s]'] /= 60
    data['averageTransferToFiringDuration[s]'] /= 60

# Wybierz odpowiednie kolumny w zależności od danych
if date == "all":
    if mode == "amount":
        y_label_intervention = "amountOfInterventions"
        y_label_firing = "amountOfFirings"
    if mode == "duration":
        y_label_intervention = "interventionsDuration[min]"
        y_label_firing = "firingsDuration[min]"
    if mode == "averageDuration":
        y_label_intervention = "averageInterventionDuration[min]"
        y_label_firing = "averageFiringDuration[min]"
    if mode == "averageTransfer":
        y_label_intervention = "averageTransferToInterventionDuration[s]"
        y_label_firing = "averageTransferToFiringDuration[s]"
elif date == "firing":
    if mode == "amount":
        y_label_firing = "amountOfFirings"
    if mode == "duration":
        y_label_firing = "firingsDuration[min]"
    if mode == "averageDuration":
        y_label_firing = "averageFiringDuration[min]"
    if mode == "averageTransfer":
        y_label_firing = "averageTransferToFiringDuration[s]"
elif date == "intervention":
    if mode == "amount":
        y_label_intervention = "amountOfInterventions"
        y_label_firing = "amountOfInterventions"
    if mode == "duration":
        y_label_intervention = "interventionsDuration[min]"
        y_label_firing = "interventionsDuration[min]"
    if mode == "averageDuration":
        y_label_intervention = "averageInterventionDuration[min]"
        y_label_firing = "averageInterventionDuration[min]"
    if mode == "averageTransfer":
        y_label_intervention = "averageTransferToInterventionDuration[s]"
        y_label_firing = "averageTransferToInterventionDuration[s]"

x_label = "Per Hour[Lp]"

# Wygeneruj wykres słupkowy
if date == "all":
    plt.bar(data.index + 1, data[y_label_firing], label="Firing")
    plt.bar(data.index + 1, data[y_label_intervention], label="Intervention", alpha=0.7)
else:
    plt.bar(data.index + 1, data[y_label_firing] if date == "firing" else data[y_label_intervention])

plt.xlabel(x_label)
plt.ylabel(y_label_firing.replace("[min]", " [h]") if mode != "averageTransfer" else y_label_firing.replace("[s]", " [min]") if date == "firing" else y_label_intervention)
plt.title(f"{mode.capitalize()} {date} per Hour")
plt.legend()

# Dostosuj etykiety na osi x
plt.xticks(range(1, len(data) + 1, 1))
plt.show()