from tensorflow.keras.models import load_model, save_model

# 1. Load the existing model
model = load_model("Skin Cancer.h5", compile=False)

# 2. Save again in proper format with no batch_input_shape issue
model.save("skin_cancer_fixed.h5")
