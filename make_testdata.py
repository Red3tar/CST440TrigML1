"""
CST-440 Lecture 3 - Test dataset for the trig model

Builds testdata.csv: a fixed set of angles the model is evaluated on AFTER training.
None of these rows are used by modelRAD.py, so they are a fair check of the model.

Columns: category, degrees, x (radians, wrapped into 0..2*pi), sin_x, cos_x

Categories
  standard  - the common unit-circle angles (0, 30, 45, 60, 90, ... 360)
  off_grid  - angles halfway between the instructor's 15-degree points (7.5, 22.5, ...)
  near_edge - angles just inside the ends of the training range (0 and 2*pi)
  wrap      - negative angles and angles past 360 (the app wraps these into 0..360)
  random    - random angles across one full turn (fixed seed, so the file is repeatable)

Run:  python make_testdata.py
Then: python app.py --data testdata.csv
"""
import numpy as np

OUT_PATH = "testdata.csv"
N_RANDOM = 50
SEED = 440                # different from the training seed (42) so the random angles differ


def build_rows():
    rng = np.random.default_rng(SEED)
    groups = {
        "standard": [0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330, 360],
        "off_grid": list(np.arange(7.5, 360, 15.0)),
        "near_edge": [0.1, 0.5, 1, 2, 358, 359, 359.5, 359.9],
        "wrap": [-30, -90, -180, -270, -360, -405, 390, 450, 540, 720, 1000],
        "random": sorted(np.round(rng.uniform(0, 360, N_RANDOM), 2)),
    }
    rows = []
    for category, degrees in groups.items():
        for deg in degrees:
            rad = float(np.deg2rad(deg) % (2 * np.pi))
            rows.append((category, float(deg), rad, np.sin(rad), np.cos(rad)))
    return rows


def main():
    rows = build_rows()
    with open(OUT_PATH, "w") as f:
        f.write("category,degrees,x,sin_x,cos_x\n")
        for category, deg, rad, s, c in rows:
            f.write(f"{category},{deg:g},{rad:.6f},{s:.6f},{c:.6f}\n")
    print(f"Saved {len(rows)} test rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
