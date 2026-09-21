"""
CST-440 Lecture 3 - Trig function ML model (angles in RADIANS): predicts sin(x) AND cos(x)

Covers Steps 1-3 of the assignment plus the export needed for Steps 4-6.

Step 1  Design   : 1 input (angle, radians) -> 1 hidden layer (ReLU) -> 2 outputs (sin, cos; linear)
Step 2  Build    : Keras Sequential model, 16 hidden neurons (slides say 8-16)
Data             : the instructor's CSV (trigdataRAD.csv) + numpy-generated angles, combined,
                   shuffled and split 70/15/15 into train / validation / test
Step 3  Train    : MSE loss, Adam, small batch size, early stopping, loss/MAE plots
Export           : trigdata_combined.csv (columns x, sin_x, cos_x), trig_model.keras,
                   trig_model.tflite (float32), trig_model_int8.tflite (quantized),
                   model_data.h (C array for Arduino)

Run:  python modelRAD.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------------------------------------------------------
# Step 1 - Design choices
# ----------------------------------------------------------------------------
HIDDEN_UNITS = 16         # slides: 8-16 neurons keeps the model Arduino-sized
N_GENERATED = 2000        # angles generated with numpy over one full period (added to the CSV rows)
TRAIN_FRAC = 0.70         # train / validation / test split of the combined dataset
VAL_FRAC = 0.15           # (the remaining 15% is the held-out test set)
BATCH_SIZE = 32           # small batch size
MAX_EPOCHS = 500          # early stopping normally ends training well before this
PATIENCE = 30
SEED = 42

CSV_PATH = "trigdataRAD.csv"             # 15-degree reference table from the instructor
COMBINED_PATH = "trigdata.csv"  # combined dataset written out: x, sin_x, cos_x
KERAS_PATH = "trig_model.keras"
TFLITE_PATH = "trig_model.tflite"
TFLITE_INT8_PATH = "trig_model_int8.tflite"
HEADER_PATH = "model_data.h"
PLOT_PATH = "training_history.png"

TWO_PI = 2 * np.pi


def trig(x):
    """Targets for the network: column 0 = sin(x), column 1 = cos(x)."""
    return np.column_stack([np.sin(x), np.cos(x)])


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def generate_data(rng):
    """Random angles over one full period and their (sin, cos) values."""
    x = rng.uniform(0.0, TWO_PI, N_GENERATED).astype(np.float32)
    return x, trig(x).astype(np.float32)


def load_csv_data():
    """The instructor's 15-degree table (columns: x, sin_x, cos_x)."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
    data = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)
    return data[:, 0].astype(np.float32), data[:, 1:3].astype(np.float32)


def save_combined_csv(x, y, path=COMBINED_PATH):
    """Write the combined dataset (CSV rows + generated rows), sorted by angle."""
    order = np.argsort(x)
    with open(path, "w") as f:
        f.write("x,sin_x,cos_x\n")
        for xi, (s, c) in zip(x[order], y[order]):
            f.write(f"{xi:.6f},{s:.6f},{c:.6f}\n")
    print(f"Saved combined dataset ({len(x)} rows) to {path}")


def build_dataset(rng):
    """
    Combine the CSV rows with the generated rows, shuffle everything together, and split
    into train / validation / test.
      train      - what the weights are fitted on
      validation - watched during training for early stopping
      test       - never touched until the final evaluation
    Returns three (X, y) pairs: X shaped (n, 1) and y shaped (n, 2) = [sin, cos].
    """
    x_csv, y_csv = load_csv_data()
    x_gen, y_gen = generate_data(rng)

    x = np.concatenate([x_csv, x_gen])
    y = np.concatenate([y_csv, y_gen])
    is_csv = np.array([True] * len(x_csv) + [False] * len(x_gen))
    save_combined_csv(x, y)

    order = rng.permutation(len(x))
    x, y, is_csv = x[order], y[order], is_csv[order]

    n = len(x)
    n_train, n_val = int(TRAIN_FRAC * n), int(VAL_FRAC * n)
    bounds = {"train": (0, n_train), "val": (n_train, n_train + n_val), "test": (n_train + n_val, n)}

    print(f"Dataset: {len(x_csv)} CSV + {len(x_gen)} generated = {n} rows")
    for name, (lo, hi) in bounds.items():
        print(f"  {name:<5} {hi - lo:>5} rows ({is_csv[lo:hi].sum()} from CSV)")

    def part(name):
        lo, hi = bounds[name]
        return x[lo:hi].reshape(-1, 1), y[lo:hi]

    return part("train"), part("val"), part("test")


# ----------------------------------------------------------------------------
# Step 2 - Build the model
# ----------------------------------------------------------------------------
def build_model():
    """
    Angle -> Dense(16, ReLU) -> Dense(2, linear) -> [predicted sin(x), predicted cos(x)]

    Initialisation matters here. Keras' defaults (random weights, zero biases) put every
    ReLU "kink" at x = 0, and because our inputs are all positive, every neuron with a
    negative weight is dead from the first step. Roughly half the hidden layer never
    trains and the fit is poor. So the hidden layer starts with its kinks spread evenly
    across [0, 2*pi]; training then only has to fine-tune them.
    """
    keras.utils.set_random_seed(SEED)
    rng = np.random.default_rng(SEED)

    model = keras.Sequential([
        layers.Input(shape=(1,)),                                  # angle in radians
        layers.Dense(HIDDEN_UNITS, activation="relu", name="hidden"),
        layers.Dense(2, activation="linear", name="output"),       # linear: outputs can be negative
    ])

    w = rng.choice([-1.0, 1.0], HIDDEN_UNITS) * rng.uniform(0.5, 1.5, HIDDEN_UNITS)
    kinks = np.linspace(0.0, TWO_PI, HIDDEN_UNITS + 2)[1:-1]
    b = -w * kinks
    model.get_layer("hidden").set_weights(
        [w.reshape(1, HIDDEN_UNITS).astype(np.float32), b.astype(np.float32)]
    )

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# ----------------------------------------------------------------------------
# Step 3 - Train
# ----------------------------------------------------------------------------
def train(model, X_train, y_train, X_val, y_val):
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )
    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        callbacks=[early_stop],
        verbose=2,
    )


def plot_history(history, path=PLOT_PATH):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(history.history["loss"], label="train")
    ax1.plot(history.history["val_loss"], label="validation")
    ax1.set_yscale("log")
    ax1.set(title="Loss (MSE)", xlabel="Epoch", ylabel="MSE")
    ax1.legend()

    ax2.plot(history.history["mae"], label="train")
    ax2.plot(history.history["val_mae"], label="validation")
    ax2.set_yscale("log")
    ax2.set(title="Mean absolute error (sin and cos)", xlabel="Epoch", ylabel="MAE")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {path}")


def report(model, X, y_true, label):
    """Per-function MAE, worst-case error, and overall accuracy = 1 - mean(distance) / 2,
    where distance is the Euclidean error between the predicted and true (sin, cos) pair
    (both are in [-1, 1], so the largest possible distance is 2)."""
    err = y_true - model.predict(X, verbose=0)
    distance = np.linalg.norm(err, axis=1)
    accuracy = 1.0 - distance.mean() / 2.0
    print(f"{label:<16} MAE sin {np.abs(err[:, 0]).mean():.5f} | MAE cos {np.abs(err[:, 1]).mean():.5f} | "
          f"max error {np.abs(err).max():.5f} | accuracy {accuracy * 100:.2f}%")


# ----------------------------------------------------------------------------
# Export - TensorFlow Lite (Step 4) and C array for the Arduino (Step 6)
# ----------------------------------------------------------------------------
def convert_to_tflite(model):
    """Return (float32 model bytes, int8-quantized model bytes)."""
    float_bytes = tf.lite.TFLiteConverter.from_keras_model(model).convert()

    def representative_data():
        # Sweep the whole input range so the quantizer sees 0 and 2*pi
        for angle in np.linspace(0.0, TWO_PI, 200, dtype=np.float32):
            yield [np.array([[angle]], dtype=np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    int8_bytes = converter.convert()
    return float_bytes, int8_bytes


def write_c_header(model_bytes, path=HEADER_PATH):
    """Dump a .tflite file as a C array (same layout as the Arduino TFLite examples)."""
    rows = []
    for i in range(0, len(model_bytes), 12):
        rows.append("  " + ", ".join(f"0x{b:02x}" for b in model_bytes[i:i + 12]))
    body = ",\n".join(rows)
    with open(path, "w") as f:
        f.write("// Auto-generated by modelRAD.py - do not edit by hand\n")
        f.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n")
        f.write(f"alignas(8) const unsigned char g_model[] = {{\n{body}\n}};\n")
        f.write(f"const unsigned int g_model_len = {len(model_bytes)};\n\n")
        f.write("#endif  // MODEL_DATA_H\n")
    print(f"Saved C array ({len(model_bytes)} bytes of model) to {path}")


# ----------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = build_dataset(rng)
    print("Outputs: sin(x) and cos(x), x in radians\n")

    model = build_model()
    model.summary()

    history = train(model, X_train, y_train, X_val, y_val)
    plot_history(history)

    print("\n=== Model Evaluation (RAD) ===")
    report(model, X_train, y_train, "Train")
    report(model, X_val, y_val, "Validation")
    report(model, X_test, y_test, "Test (held out)")

    model.save(KERAS_PATH)
    print(f"\nSaved Keras model to {KERAS_PATH}")

    float_bytes, int8_bytes = convert_to_tflite(model)
    with open(TFLITE_PATH, "wb") as f:
        f.write(float_bytes)
    with open(TFLITE_INT8_PATH, "wb") as f:
        f.write(int8_bytes)
    print(f"Saved {TFLITE_PATH} ({len(float_bytes)} bytes) and "
          f"{TFLITE_INT8_PATH} ({len(int8_bytes)} bytes)")

    write_c_header(int8_bytes)


if __name__ == "__main__":
    main()
