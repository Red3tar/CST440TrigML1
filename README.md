# CST-440 Trig Function ML Model

A small neural network that takes an angle and predicts **sin(x)** and **cos(x)**. It is
trained in Python with TensorFlow/Keras, converted to TensorFlow Lite, and exported as a
C array so it can run on an Arduino.

## What's in this repo

| File | What it is |
|---|---|
| `modelRAD.py` | Trains the model (Steps 1-3) and exports the `.keras`, `.tflite` and `model_data.h` files |
| `app.py` | Runs the trained model: interactive mode, a test report, and scoring on a test dataset (Steps 4-5) |
| `make_testdata.py` | Builds `testdata.csv`, the test dataset |
| `trigdataRAD.csv` | Instructor's reference table: one row every 15 degrees (input for training) |
| `trigdata.csv` | Combined training data (**written by `modelRAD.py`, don't edit by hand**) |
| `testdata.csv` | 110 test angles the model never trains on (made by `make_testdata.py`) |
| `trig_model.keras` / `trig_model.tflite` / `trig_model_int8.tflite` | Trained model: Keras, float32 TFLite, int8 (quantized) TFLite |
| `model_data.h` | The int8 model as a C array for the Arduino sketch |
| `training_history.png` | Loss / error curves from training |
| `trig_model_report.ipynb` | Notebook write-up of the project |
| `requirements.txt` | Python packages to install |

The trained model files are already committed, so **you don't have to train the model
to run the app**. Retrain only if you change `modelRAD.py` or the training data.

---

## Installation

### 1. Install Python 3.12

TensorFlow only supports certain Python versions. This project was tested with
**Python 3.12**. Python 3.10, 3.11 and 3.13 should also work. **Don't use 3.14 or newer**,
because TensorFlow may not install.

- **Windows:** download Python 3.12 from <https://www.python.org/downloads/>. In the
  installer, tick **"Add python.exe to PATH"**.
- **macOS:** `brew install python@3.12`, or use the python.org installer.

Check that it installed:

```bash
# Windows
py -3.12 --version

# macOS / Linux
python3.12 --version
```

### 2. Get the code

```bash
git clone https://github.com/lukehoyle2004/CST-440-Clone.git
cd CST-440-Clone
```

(You can also use **GitHub Desktop → Clone repository**, then open the folder in VS Code.)

### 3. Create a virtual environment

A virtual environment (`.venv`) keeps this project's packages separate from the rest of
your computer. Run these commands from inside the project folder.

**Windows (PowerShell or the VS Code terminal):**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
```

If PowerShell says *"running scripts is disabled on this system"*, run the line below
once, then run `activate` again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**macOS / Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

When the environment is active, your terminal prompt starts with `(.venv)`.

### 4. Install the packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow is a large download (about 300-500 MB), so this can take a few minutes.

### 5. Check that it works

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

This should print `2.21.0`. It may also print some `oneDNN` / `absl` info lines first.
Those are harmless.

### Using VS Code

1. Install the **Python** extension (by Microsoft).
2. Open the project folder: **File → Open Folder**.
3. Press `Ctrl+Shift+P` (`Cmd+Shift+P` on Mac), choose **Python: Select Interpreter**, and
   pick the one that shows `.venv`.
4. Open a new terminal (**Terminal → New Terminal**). It activates `.venv` automatically.
5. To open the notebook, select the `.venv` kernel in the top right. VS Code may ask to
   install `ipykernel`; click Install.

---

## Running the project

Run all of these from the project folder with `.venv` active.

### Try the model (interactive)

```bash
python app.py
```

Type an angle in **degrees** (for example `45`, `270`, `-90`). The app shows the
predicted and true sin/cos values, the error and the inference time. Type `exit` to quit.

### Test report (edge cases + 1000-angle sweep)

```bash
python app.py --test
```

### Score the model on the test dataset

```bash
python make_testdata.py              # (re)builds testdata.csv
python app.py --data testdata.csv
```

This prints the error and accuracy for each category of test angle, then the 5 worst
predictions. To test the quantized model that goes on the Arduino:

```bash
python app.py --model trig_model_int8.tflite --data testdata.csv
```

### Retrain the model (optional)

```bash
python modelRAD.py
```

Training takes about a minute on a laptop. It **overwrites** `trigdata.csv`, the three
`trig_model.*` files, `model_data.h` and `training_history.png`. After retraining, copy
the new `model_data.h` into the Arduino sketch.

---

## The test dataset

`testdata.csv` has 110 rows with the columns `category, degrees, x, sin_x, cos_x`. `x` is the
angle in radians, wrapped into 0..2π. None of these rows are used for training, so the
test is fair.

| Category | Angles | Why it's there |
|---|---|---|
| `standard` | 0, 30, 45, 60, 90 … 360 | Common unit-circle angles |
| `off_grid` | 7.5, 22.5, 37.5 … 352.5 | Halfway between the instructor's 15° points |
| `near_edge` | 0.1, 0.5, 1, 2, 358, 359, 359.5, 359.9 | Ends of the training range, where the model is weakest |
| `wrap` | -30, -90, -360, 450, 720, 1000 … | Angles outside 0-360; the app wraps them into range first |
| `random` | 50 random angles (seed 440) | General coverage; the same angles every run |

To add or change test points, edit the lists in `build_rows()` in `make_testdata.py`,
then run it again.

### Current results

| Model | Accuracy on test data | Largest error |
|---|---|---|
| `trig_model.tflite` (float32) | 99.28% | 0.033 |
| `trig_model_int8.tflite` (int8) | 99.18% | 0.042 |

Accuracy = 1 − (average distance between the predicted and true (sin, cos) point) / 2.
The largest errors are at 0°/360°, where the model predicts cos ≈ 1.033.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No matching distribution found for tensorflow` | Your Python version isn't supported. Delete `.venv` and recreate it with Python 3.12 (step 3). |
| `ModuleNotFoundError: No module named 'tensorflow'` | The virtual environment isn't active, or VS Code is using the wrong interpreter. Activate `.venv` (step 3) or reselect the interpreter. |
| `Model file not found: trig_model.tflite` | Run the command from the project folder, or run `python modelRAD.py` to rebuild the model. |
| `Test data not found: testdata.csv` | Run `python make_testdata.py`. |
| Lots of `oneDNN`, `absl` or `deprecated` messages | These are warnings, not errors. To hide them, set `TF_CPP_MIN_LOG_LEVEL=2` before running. |
| Windows: install fails with a "long path" error | Enable long paths (search "Enable Win32 long paths"), or clone the repo to a shorter path such as `C:\dev\CST-440-Clone`. |
