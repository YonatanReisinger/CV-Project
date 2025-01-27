from Experiment import Experiment
import os

def load_experiments(directory=".", starts_with="Experiment"):
    pkl_files = [f for f in os.listdir(directory) if f.endswith(".pkl") and f.startswith(starts_with)]
    loaded_data = {}

    for pkl_file in pkl_files:
        file_path = os.path.join(directory, pkl_file)
        try:
            loaded_data[pkl_file] = Experiment.from_pickle(file_path)
            print(loaded_data[pkl_file])
        except Exception as e:
            print(f"Failed to load {pkl_file}: {e}")

    return loaded_data

def load_experiments_and_print():
    loaded_data = load_experiments()
    for name, data in loaded_data.items():
        print(data)