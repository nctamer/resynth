import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


names = ["DanclaOp84-BernardChevalier",
         "KayserOp20-FabricioValvasori",
         "KayserOp20-PatrickRafferty",
         "KayserOp20-AlexandrosIakovou",
         "Kreutzer42-CihatAskin",
         "MazasOp36-BernardChevalier",
         "WohlfahrtOp45-PatrickRafferty"]

store_patterns = np.zeros((0, 15))
n_total = 0
n_valid = 0

parent_folder = "/home/nazif/PycharmProjects/data"

for name in names:
    main_path = os.path.join(parent_folder, name)

    save_file_path = os.path.join(main_path, "analyzed")

    for file in sorted(os.listdir(save_file_path)):
        with open(os.path.join(save_file_path, file), "rb") as f:
            analyzed = pkl.load(f)
            instrument_pattern = analyzed["harmonic"]["mag"] + 100
            valid_bool = instrument_pattern[:, 0] > 0
            instrument_pattern = instrument_pattern[valid_bool, :15]
            instrument_pattern /= instrument_pattern[:, 0][:, None]
            n_total += len(valid_bool)
            n_valid += sum(valid_bool)
            store_patterns = np.vstack((store_patterns, instrument_pattern))

    print(name)
    np.save(parent_folder, store_patterns)

print("total #samples: ", n_total)
print("samples with f0err<5: ", n_valid)
print("coverage (%): ", 100*(n_valid/n_total))
