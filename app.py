"""
CST-440 Lecture 3 - Trig function application (Steps 4-5)

Loads the TensorFlow Lite model made by modelRAD.py, takes an angle in DEGREES from the
user, runs inference, and shows the predicted sin(x) and cos(x) next to the true values.

  python app.py                                 # interactive (enter angles in degrees)
  python app.py --test                          # edge cases + accuracy/size/speed report
  python app.py --model trig_model_int8.tflite  # try the quantized model
  python app.py --data testdata.csv             # score on the test dataset (make_testdata.py)
"""
import argparse
import os
import time

import numpy as np
import tensorflow as tf

DEFAULT_MODEL = "trig_model.tflite"
TWO_PI = 2 * np.pi


class TrigModel:
    """Thin wrapper around a .tflite model (handles float32 and int8 models)."""

    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path} (run modelRAD.py first)")
        self.path = path
        self.size_bytes = os.path.getsize(path)
        self.interpreter = tf.lite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]

    def predict(self, angle_rad):
        """Return (sin, cos) predicted for an angle in radians (must be within 0..2*pi)."""
        x = np.float32(angle_rad)
        if self.inp["dtype"] == np.int8:                 # quantize the input
            scale, zero_point = self.inp["quantization"]
            x = np.clip(np.round(x / scale + zero_point), -128, 127)
        self.interpreter.set_tensor(self.inp["index"], np.array([[x]], dtype=self.inp["dtype"]))
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.out["index"])[0].astype(np.float32)
        if self.out["dtype"] == np.int8:                 # dequantize the output
            scale, zero_point = self.out["quantization"]
            y = (y - zero_point) * scale
        return float(y[0]), float(y[1])


def true_values(angle_rad):
    return float(np.sin(angle_rad)), float(np.cos(angle_rad))


def accuracy_pct(pred, true):
    """Same metric as modelRAD.py: 1 - distance / 2, where distance is the Euclidean error
    between the predicted and true (sin, cos) pair (the largest possible distance is 2)."""
    distance = np.hypot(pred[0] - true[0], pred[1] - true[1])
    return (1.0 - distance / 2.0) * 100.0


def wrap_angle(angle_rad):
    """The model only saw 0..2*pi, so reduce any angle into that range first."""
    return float(angle_rad % TWO_PI)


# ----------------------------------------------------------------------------
# Step 4 - interactive application
# ----------------------------------------------------------------------------
def run_interactive(model):
    print("\n=== Trig Function Approximation: sin(x) and cos(x) ===")
    print("Enter angles in DEGREES. Type 'exit' to quit.\n")

    while True:
        text = input("Angle (degrees): ").strip()
        if text.lower() == "exit":
            break
        try:
            angle_deg = float(text)
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        angle_rad = np.deg2rad(angle_deg)      # the model works in radians internally
        wrapped = wrap_angle(angle_rad)

        start = time.perf_counter()
        pred_sin, pred_cos = model.predict(wrapped)
        elapsed_ms = (time.perf_counter() - start) * 1000
        true_sin, true_cos = true_values(wrapped)

        print("\n=== Result ===")
        print(f"Angle (degrees):     {angle_deg}")
        if not np.isclose(wrapped, angle_rad):
            print(f"Wrapped into 0-360:  {np.rad2deg(wrapped):.4f} degrees")
        print(f"Angle (radians):     {wrapped:.6f}")
        print(f"Predicted sin(x):    {pred_sin:.6f}")
        print(f"Actual sin(x):       {true_sin:.6f}")
        print(f"Predicted cos(x):    {pred_cos:.6f}")
        print(f"Actual cos(x):       {true_cos:.6f}")
        print(f"Absolute error:      sin {abs(pred_sin - true_sin):.6f} | cos {abs(pred_cos - true_cos):.6f}")
        print(f"Model accuracy:      {accuracy_pct((pred_sin, pred_cos), (true_sin, true_cos)):.2f}%")
        print(f"Inference time:      {elapsed_ms:.3f} ms (on this computer)\n")


# ----------------------------------------------------------------------------
# Step 5 - test the application
# ----------------------------------------------------------------------------
def run_tests(model):
    print(f"\n=== Test report: {model.path} ===")
    print(f"Model size: {model.size_bytes} bytes\n")

    print("Edge cases:")
    print(f"{'Angle (deg)':>12} | {'sin pred':>9} {'sin true':>9} {'error':>8} | "
          f"{'cos pred':>9} {'cos true':>9} {'error':>8}")
    for deg in [0, 30, 45, 90, 180, 270, 360, 450, -90]:   # last two check the wrap-around
        rad = wrap_angle(np.deg2rad(deg))
        (ps, pc), (ts, tc) = model.predict(rad), true_values(rad)
        print(f"{deg:>12} | {ps:>9.5f} {ts:>9.5f} {abs(ps - ts):>8.5f} | "
              f"{pc:>9.5f} {tc:>9.5f} {abs(pc - tc):>8.5f}")

    angles = np.linspace(0.0, TWO_PI, 1000)
    start = time.perf_counter()
    preds = np.array([model.predict(a) for a in angles])
    per_call_ms = (time.perf_counter() - start) / len(angles) * 1000
    truth = np.column_stack([np.sin(angles), np.cos(angles)])
    errors = np.abs(preds - truth)
    distance = np.linalg.norm(preds - truth, axis=1)
    worst = np.rad2deg(angles[errors.max(axis=1).argmax()])

    print("\nSweep of 1000 angles across 0..2pi:")
    print(f"  MAE sin:            {errors[:, 0].mean():.5f}")
    print(f"  MAE cos:            {errors[:, 1].mean():.5f}")
    print(f"  Max error:          {errors.max():.5f} (at {worst:.1f} deg)")
    print(f"  Accuracy:           {(1 - distance.mean() / 2) * 100:.2f}%")
    print(f"  Avg inference time: {per_call_ms:.4f} ms per call (on this computer)")


def run_dataset(model, path):
    """Score the model on a CSV made by make_testdata.py (category, degrees, x, sin_x, cos_x)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test data not found: {path} (run make_testdata.py first)")
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")

    preds = np.array([model.predict(x) for x in data["x"]])
    truth = np.column_stack([data["sin_x"], data["cos_x"]])
    errors = np.abs(preds - truth)
    distance = np.linalg.norm(preds - truth, axis=1)

    print(f"\n=== Test dataset: {path} ({len(data)} rows) on {model.path} ===")
    print(f"{'Category':<10} {'rows':>5} | {'MAE sin':>8} {'MAE cos':>8} {'max err':>8} | {'accuracy':>8}")
    categories = list(dict.fromkeys(data["category"]))   # keep file order
    for cat in categories + ["ALL"]:
        m = np.ones(len(data), bool) if cat == "ALL" else data["category"] == cat
        print(f"{cat:<10} {m.sum():>5} | {errors[m, 0].mean():>8.5f} {errors[m, 1].mean():>8.5f} "
              f"{errors[m].max():>8.5f} | {(1 - distance[m].mean() / 2) * 100:>7.2f}%")

    print("\nWorst 5 rows:")
    for i in np.argsort(errors.max(axis=1))[::-1][:5]:
        print(f"  {data['category'][i]:<10} {data['degrees'][i]:>8g} deg | "
              f"sin {preds[i, 0]:>8.5f} vs {truth[i, 0]:>8.5f} | cos {preds[i, 1]:>8.5f} vs {truth[i, 1]:>8.5f}")


def main():
    parser = argparse.ArgumentParser(description="Trig function ML application (sin and cos)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="path to a .tflite model")
    parser.add_argument("--test", action="store_true", help="run the edge-case/accuracy report")
    parser.add_argument("--data", help="score the model on a test CSV (see make_testdata.py)")
    args = parser.parse_args()

    model = TrigModel(args.model)
    print(f"Loaded model: {args.model}")

    if args.test:
        run_tests(model)
    if args.data:
        run_dataset(model, args.data)
    if not (args.test or args.data):
        run_interactive(model)


if __name__ == "__main__":
    main()
