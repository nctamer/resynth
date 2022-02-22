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

pitch_track_deviation = np.array([])
n_total = 0
n_valid = 0

for name in names:
    main_path = os.path.join("/home/nazif/PycharmProjects/data", name)

    save_file_path = os.path.join(main_path, "analyzed")

    for file in sorted(os.listdir(save_file_path)):
        with open(os.path.join(save_file_path, file), "rb") as f:
            analyzed = pkl.load(f)
            error = analyzed["f0"]["error"]
            valid_bool = (error < 0.9) * (analyzed["f0"]["old"] > 0) * (analyzed["f0"]["new"] > 0)
            n_total += len(valid_bool)
            n_valid += sum(valid_bool)
            old = analyzed["f0"]["old"][valid_bool]
            new = analyzed["f0"]["new"][valid_bool]
            deviation_rate = new/old
            deviation_cents = np.log2(deviation_rate)*1200
            pitch_track_deviation = np.hstack((pitch_track_deviation, deviation_cents))
    print(name)

print("total #samples: ", n_total)
print("samples with f0err<5: ", n_valid)
print("coverage (%): ", 100*(n_valid/n_total))
print("mean of deviation (cents): ", np.mean(pitch_track_deviation))
print("std of deviation (cents): ", np.std(pitch_track_deviation))
plt.hist(pitch_track_deviation, bins=41)
plt.xlabel("Deviation btw old and new pitch track (cents)")
plt.ylabel("Frame count")
plt.title("mean: {:.3f}, std: {:.3f}".format(np.mean(pitch_track_deviation), np.std(pitch_track_deviation)))
plt.show()
