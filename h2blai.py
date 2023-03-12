import numpy as np
open("facepoints.blai", "wb").write(np.array(list(map(eval, open("model.h").read().split("static const uint8_t blai_model_bin[] = {")[1].split("};")[0].replace(" ", "").replace("\n", "").split(","))), 'uint8'))
