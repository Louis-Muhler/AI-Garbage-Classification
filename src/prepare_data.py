import splitfolders
import os

def split_data(input_folder='data', output_folder='data_split'):
    """
    Teilt die Daten systematisch auf. 
    Ratio: 80% Training, 10% Validierung, 10% Test.
    """
    if not os.path.exists(output_folder):
        splitfolders.ratio(input_folder, 
                           output=output_folder, 
                           seed=42, 
                           ratio=(.8, .1, .1), 
                           group_prefix=None)
        print(f"Daten erfolgreich in {output_folder} aufgeteilt.")
    else:
        print("Ordner existiert bereits. Kein neuer Split notwendig.")
