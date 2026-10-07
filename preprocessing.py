"""
preprocess.py - merge three Kaggle datasets into one meta stroke dataset.

Usage:
    python preprocess.py             # build the composite CSVs

Output schema: age, sex (1=male, 0=female), bmi, ever_smoked, heart_disease,
hypertension, stroke, source. `source` is not a model feature, but we kept it so
splits and metrics can be broken down per dataset.

Two files are written: all sources, and the same data without `stroke_prediction`.
That dataset's label is close to random (a logistic regression on it scores
AUC ~0.51), so the second file lets us measure what it adds or costs.
"""
import glob
import os

import pandas as pd

RAW_DIR = "StrokeData"
OUT_PATH = "StrokeData/stroke_composite.csv"
OUT_PATH_NO_SP = "StrokeData/stroke_composite_no_stroke_prediction.csv"
NOISY_SOURCE = "stroke_prediction"

# File patterns for the raw Kaggle downloads
HEALTHCARE_GLOB = "*healthcare-dataset-stroke*.csv"
STROKE_PRED_GLOB = "stroke_prediction_dataset.csv"
DIABETES_GLOB = "diabetes_data.csv"

# Column names in the stroke_prediction file.
STROKE_PRED_COLS = {
    "age": "Age",
    "sex": "Gender",
    "bmi": "Body Mass Index (BMI)",
    "smoking": "Smoking Status",
    "heart_disease": "Heart Disease",
    "hypertension": "Hypertension",
    "bp": "Blood Pressure Levels",
    "glucose": "Average Glucose Level",
    "stroke": "Diagnosis",
}

# The diabetes_data file (BRFSS survey) stores age as categories 1-13, each a 5-year band
# (1 = 18-24, 2 = 25-29, ... 13 = 80+). We use each band's midpoint.
DIABETES_AGE_MIDPOINTS = {1: 21, 2: 27, 3: 32, 4: 37, 5: 42, 6: 47, 7: 52,
                       8: 57, 9: 62, 10: 67, 11: 72, 12: 77, 13: 82}

# Only diabetes_data has these. The other sources get age-band averages (see main).
EXTRA_COLS = ["gen_health", "diff_walking", "high_chol"]

# diabetes = 1 for a glucose reading at or above the standard diabetes cutoff (mg/dL)
GLUCOSE_CUTOFF = 126

FINAL_COLS = ["age", "sex", "bmi", "ever_smoked", "heart_disease", "hypertension",
              "diabetes"] + EXTRA_COLS + ["stroke", "source"]


def find_one(pattern):
    matches = glob.glob(os.path.join(RAW_DIR, pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file matching {pattern!r} in {RAW_DIR}, got {matches}")
    return matches[0]


def sex_to_int(series):
    """Male -> 1, female -> 0. Anything else (e.g. 'Other') becomes NaN and is dropped later."""
    s = series.astype(str).str.strip().str.lower()
    return s.map({"male": 1, "female": 0})


def load_healthcare_stroke():
    df = pd.read_csv(find_one(HEALTHCARE_GLOB))
    smoke = df["smoking_status"].astype(str).str.strip().str.lower()
    ever = smoke.map({"formerly smoked": 1, "smokes": 1, "never smoked": 0})
    return pd.DataFrame({
        "age": pd.to_numeric(df["age"], errors="coerce"),
        "sex": sex_to_int(df["gender"]),
        "bmi": pd.to_numeric(df["bmi"], errors="coerce"),  # "N/A" strings -> NaN
        "ever_smoked": ever,
        "heart_disease": df["heart_disease"].astype(int),
        "hypertension": df["hypertension"].astype(int),
        "diabetes": (pd.to_numeric(df["avg_glucose_level"], errors="coerce") >= GLUCOSE_CUTOFF).astype(int),
        "stroke": df["stroke"].astype(int),
        "source": "healthcare_stroke",
    })


def load_stroke_prediction():
    c = STROKE_PRED_COLS
    df = pd.read_csv(find_one(STROKE_PRED_GLOB))

    smoke = df[c["smoking"]].astype(str).str.strip().str.lower()
    ever = smoke.map({"currently smokes": 1, "formerly smoked": 1, "non-smoker": 0})

    # Derive hypertension from the readings

    bp = df[c["bp"]].astype(str).str.extract(r"(\d+)\s*/\s*(\d+)").astype(float)
    sys_bp, dia_bp = bp[0], bp[1]
    hyp_from_bp = ((sys_bp >= 140) | (dia_bp >= 90)).astype(float)
    hypertension = hyp_from_bp.where(sys_bp.notna(), pd.to_numeric(df[c["hypertension"]]))

    out = pd.DataFrame({
        "age": pd.to_numeric(df[c["age"]], errors="coerce"),
        "sex": sex_to_int(df[c["sex"]]),
        "bmi": pd.to_numeric(df[c["bmi"]], errors="coerce"),
        "ever_smoked": ever,
        "heart_disease": pd.to_numeric(df[c["heart_disease"]], errors="coerce"),
        "hypertension": hypertension,
        "diabetes": (pd.to_numeric(df[c["glucose"]], errors="coerce") >= GLUCOSE_CUTOFF).astype(int),
        "stroke": df[c["stroke"]].astype(str).str.strip().str.lower()
                    .map({"stroke": 1, "no stroke": 0}),
        "source": "stroke_prediction",
    })
    # Diastolic above systolic is physically impossible
    return out[~(dia_bp > sys_bp).reindex(out.index).fillna(False)]


def load_diabetes():
    df = pd.read_csv(find_one(DIABETES_GLOB))
    return pd.DataFrame({
        "age": df["Age"].map(DIABETES_AGE_MIDPOINTS),
        "sex": df["Sex"].astype(int),
        "bmi": pd.to_numeric(df["BMI"], errors="coerce"),
        "ever_smoked": df["Smoker"].astype(int),
        "heart_disease": df["HeartDiseaseorAttack"].astype(int),
        "hypertension": df["HighBP"].astype(int),
        "diabetes": df["Diabetes"].astype(int),
        "gen_health": df["GenHlth"],           # 1 (excellent) to 5 (poor)
        "diff_walking": df["DiffWalk"],
        "high_chol": df["HighChol"],
        "stroke": df["Stroke"].astype(int),
        "source": "diabetes_data",
    })


def main():
    frames = [load_healthcare_stroke(), load_stroke_prediction(), load_diabetes()]
    for f in frames:
        print(f"{f['source'].iloc[0]:>17}: {len(f):>6} rows loaded, "
              f"stroke rate {f['stroke'].mean():.3f}")
    df = pd.concat(frames, ignore_index=True)[FINAL_COLS]
    print(f"\nComposite before cleaning: {len(df)} rows")


    required = ["age", "sex", "ever_smoked", "heart_disease", "hypertension", "stroke"]
    df = df.dropna(subset=required)
    df = df[df["age"] >= 18]

    for col in ["sex", "ever_smoked", "heart_disease", "hypertension", "diabetes", "stroke"]:
        df[col] = df[col].astype(int)


    band = (df["age"] // 10).astype(int)
    df["bmi"] = df["bmi"].fillna(df.groupby(band)["bmi"].transform("median"))
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())  # safety net for an empty band

    # The other sources lack the extra columns, so fill them with the diabetes_data
    # average for the same age band. Uses features only, never the stroke label.
    ref = df[df["source"] == "diabetes_data"]
    band_means = ref.groupby(band[ref.index])[EXTRA_COLS].mean()
    fill = band_means.reindex(band.values).set_axis(df.index)
    df[EXTRA_COLS] = df[EXTRA_COLS].fillna(fill).fillna(ref[EXTRA_COLS].mean())

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    df_no_sp = df[df["source"] != NOISY_SOURCE]
    df_no_sp.to_csv(OUT_PATH_NO_SP, index=False)

    print(f"Composite after cleaning:  {len(df)} rows")
    print("\nPer-source summary:")
    print(df.groupby("source").agg(rows=("stroke", "size"),
                                   stroke_rate=("stroke", "mean"),
                                   mean_age=("age", "mean")).round(3))
    print(f"\nWrote {OUT_PATH} ({len(df)} rows)")
    print(f"Wrote {OUT_PATH_NO_SP} ({len(df_no_sp)} rows)")


if __name__ == "__main__":
    main()