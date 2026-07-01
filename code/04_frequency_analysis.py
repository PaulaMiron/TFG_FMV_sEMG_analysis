from pathlib import Path
import unicodedata
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch, iirnotch, filtfilt
from scipy.integrate import trapezoid


REPO_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = REPO_DIR / "outputs"

INVENTORY_FILE = BASE_DIR / "inventory_emg_files.xlsx"
OUTPUT_FILE = BASE_DIR / "frequency_results.xlsx"
DEBUG_OUTPUT_FILE = BASE_DIR / "frequency_results_FULL.xlsx"
SAVE_DEBUG_WORKBOOK = False 

FS = 1024

LOWCUT = 20
HIGHCUT = 450
FILTER_ORDER = 4

MOVEMENT_START_SEC = 5
METRONOME_BPM = 60
DIRECTION_CHANGES_PER_CYCLE = 2
N_CENTRAL_CYCLES = 10
CYCLE_DURATION_SECONDS = DIRECTION_CHANGES_PER_CYCLE * 60 / METRONOME_BPM
CENTRAL_WINDOW_SECONDS = N_CENTRAL_CYCLES * CYCLE_DURATION_SECONDS
CENTRAL_WINDOW_SAMPLES = int(round(CENTRAL_WINDOW_SECONDS * FS))

WELCH_NPERSEG = 2048
WELCH_NOVERLAP = WELCH_NPERSEG // 2
WELCH_NFFT = 2048

LINE_NOISE_LOW = 49
LINE_NOISE_HIGH = 51

APPLY_NOTCH = True
NOTCH_FREQ = 50
NOTCH_Q = 30

POWER_BANDS = {
    "20_60": (20, 60),
    "60_90": (60, 90),
    "90_150": (90, 150),
    "150_250": (150, 250),
    "250_450": (250, 450),
}


def normalize_text(value):
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))

    return text
    
def trial_number_to_trial_cat(trial_number):
    trial_map = {
        1: "Pre",
        2: "Post",
        3: "Post5",
        4: "Post10",
        5: "Post15",
        6: "Post20",
        7: "Post25",
        8: "Post30",
    }

    return trial_map.get(int(trial_number), f"Trial{int(trial_number)}")

def read_emg_csv(csv_path):
    try:
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8", decimal=",")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, sep=";", encoding="latin1", decimal=",")

    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.strip()
            )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def interpolate_nans(x):
    x = pd.Series(x, dtype="float64")

    if x.isna().all():
        return x.to_numpy()

    x = x.interpolate(limit_direction="both")

    return x.to_numpy(dtype=float)

def apply_notch_filter(signal, fs=FS, notch_freq=NOTCH_FREQ, q=NOTCH_Q):
    b, a = iirnotch(w0=notch_freq, Q=q, fs=fs)
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal

def filter_emg_signal(signal, fs=FS, lowcut=LOWCUT, highcut=HIGHCUT, order=FILTER_ORDER):
    x = pd.to_numeric(pd.Series(signal), errors="coerce").to_numpy(dtype=float)
    x = interpolate_nans(x)

    if np.isnan(x).all():
        raise ValueError("Signal contains only NaN values")

    if len(x) < 3 * fs:
        raise ValueError("Signal too short for reliable filtering")

    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist

    if high >= 1:
        raise ValueError("Highcut frequency must be lower than Nyquist frequency")

    sos = butter(order, [low, high], btype="bandpass", output="sos")
    x_filtered = sosfiltfilt(sos, x)

    if APPLY_NOTCH:
        x_filtered = apply_notch_filter(
            x_filtered,
            fs=fs,
            notch_freq=NOTCH_FREQ,
            q=NOTCH_Q,
        )

    # Remove residual DC component
    x_filtered = x_filtered - np.mean(x_filtered)

    return x_filtered


def find_muscle_column(columns, muscle, side):
    muscle = normalize_text(muscle)
    side = normalize_text(side)

    exact_candidates = []
    fallback_candidates = []

    for col in columns:
        col_norm = normalize_text(col)

        if muscle == "biceps":
            has_muscle = "biceps" in col_norm
        elif muscle == "triceps":
            has_muscle = "triceps" in col_norm
        else:
            raise ValueError(f"Muscle not recognised: {muscle}")

        if not has_muscle:
            continue

        has_side = bool(side) and (
            f"({side})" in col_norm
            or f"[{side}]" in col_norm
            or f"_{side}_" in col_norm
            or f" {side} " in col_norm
            or col_norm.endswith(f"_{side}")
            or col_norm.endswith(f" {side}")
            or col_norm.endswith(f"({side})")
        )

        if has_side:
            exact_candidates.append(col)
        else:
            fallback_candidates.append(col)

    if len(exact_candidates) > 0:
        return exact_candidates[0]

    if len(fallback_candidates) > 0:
        return fallback_candidates[0]

    return None


def protocol_cycle_duration_sec(metronome_bpm=METRONOME_BPM, direction_changes_per_cycle=DIRECTION_CHANGES_PER_CYCLE):
    if metronome_bpm <= 0:
        raise ValueError("Metronome BPM must be greater than zero")

    return direction_changes_per_cycle * 60 / metronome_bpm


def select_central_window(n_samples, fs=FS, movement_start_sec=MOVEMENT_START_SEC, n_cycles=N_CENTRAL_CYCLES, metronome_bpm=METRONOME_BPM):
    movement_start_sample = int(round(movement_start_sec * fs))
    movement_end_sample = n_samples

    cycle_duration_sec = protocol_cycle_duration_sec(metronome_bpm)
    segment_duration_sec = n_cycles * cycle_duration_sec
    segment_samples = int(round(segment_duration_sec * fs))

    available_movement_samples = movement_end_sample - movement_start_sample

    if available_movement_samples < segment_samples:
        raise ValueError("Movement period is too short for the requested central-cycle window")

    movement_center_sample = (movement_start_sample + movement_end_sample) // 2

    segment_start_sample = movement_center_sample - segment_samples // 2
    segment_start_sample = max(
        movement_start_sample,
        min(segment_start_sample, movement_end_sample - segment_samples),
    )

    segment_end_sample = segment_start_sample + segment_samples

    return {
        "cycle_selection_method": "central_fixed_window_from_metronome_protocol",
        "cycle_selection_status": "ok",
        "expected_cycle_duration_s": cycle_duration_sec,
        "expected_cycle_frequency_hz": 1.0 / cycle_duration_sec,
        "n_cycles_selected": n_cycles,
        "segment_duration_s": segment_duration_sec,
        "segment_start_sample": segment_start_sample,
        "segment_end_sample": segment_end_sample,
        "central_window_samples": segment_samples,
        "segment_start_s": segment_start_sample / fs,
        "segment_end_s": segment_end_sample / fs,
        "recording_duration_s": n_samples / fs,
    }


def compute_band_power(freqs, psd, low, high):
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return np.nan
    return float(trapezoid(psd[mask], freqs[mask]))


def compute_mnf(freqs, psd):
    total_power = trapezoid(psd, freqs)
    if total_power <= 0 or not np.isfinite(total_power):
        return np.nan
    return float(trapezoid(freqs * psd, freqs) / total_power)


def compute_mdf(freqs, psd):
    total_power = trapezoid(psd, freqs)
    if total_power <= 0 or not np.isfinite(total_power):
        return np.nan
    cumulative_power = np.zeros_like(psd, dtype=float)

    for i in range(1, len(psd)):
        cumulative_power[i] = trapezoid(psd[: i + 1], freqs[: i + 1])

    half_power = total_power / 2
    idx = int(np.searchsorted(cumulative_power, half_power))
    idx = min(idx, len(freqs) - 1)

    return float(freqs[idx])


def compute_fpeak(freqs, psd):
    if len(freqs) == 0 or len(psd) == 0:
        return np.nan, np.nan
    idx = int(np.argmax(psd))
    return float(freqs[idx]), float(psd[idx])


def compute_fpeak_excluding_band(freqs, psd, low, high):
    mask = ~((freqs >= low) & (freqs <= high))
    freqs_clean = freqs[mask]
    psd_clean = psd[mask]

    if len(freqs_clean) == 0:
        return np.nan, np.nan

    idx = int(np.argmax(psd_clean))

    return float(freqs_clean[idx]), float(psd_clean[idx])

def compute_frequency_features(signal, segment_start_sample, segment_end_sample, fs=FS):
    x_filtered = filter_emg_signal(signal, fs=fs)

    x_segment = x_filtered[segment_start_sample:segment_end_sample]

    if len(x_segment) == 0:
        raise ValueError("Selected central window contains no EMG samples")

    if len(x_segment) < WELCH_NPERSEG:
        raise ValueError("Selected segment is too short for Welch PSD")

    x_segment = x_segment - np.mean(x_segment)

    freqs, psd = welch(
        x_segment,
        fs=fs,
        window="hann",
        nperseg=WELCH_NPERSEG,
        noverlap=WELCH_NOVERLAP,
        nfft=WELCH_NFFT,
        detrend="constant",
        scaling="density",
    )

    band_mask = (
        (freqs >= LOWCUT)
        & (freqs <= HIGHCUT)
        & np.isfinite(psd)
    )

    freqs_band = freqs[band_mask]
    psd_band = psd[band_mask]

    if len(freqs_band) == 0:
        raise ValueError("No PSD values in the EMG frequency band")

    total_power = float(trapezoid(psd_band, freqs_band))
    total_power_sum = float(np.sum(psd_band))

    mnf = compute_mnf(freqs_band, psd_band)
    mdf = compute_mdf(freqs_band, psd_band)

    fpeak, fpeak_power = compute_fpeak(freqs_band, psd_band)

    fpeak_no50, fpeak_power_no50 = compute_fpeak_excluding_band(
        freqs_band,
        psd_band,
        LINE_NOISE_LOW,
        LINE_NOISE_HIGH,
    )

    line_noise_power = compute_band_power(
        freqs_band,
        psd_band,
        LINE_NOISE_LOW,
        LINE_NOISE_HIGH,
    )

    if total_power > 0 and np.isfinite(total_power):
        line_noise_ratio = line_noise_power / total_power
    else:
        line_noise_ratio = np.nan

    features = {
        "mnf_10cycles": mnf,
        "mdf_10cycles": mdf,

        "peak_frequency_10cycles": fpeak,
        "peak_power_10cycles": fpeak_power,

        "peak_frequency_no50_10cycles": fpeak_no50,
        "peak_power_no50_10cycles": fpeak_power_no50,

        "total_power_10cycles": total_power,
        "total_power_sum_10cycles": total_power_sum,

        "line_noise_power_49_51": line_noise_power,
        "line_noise_ratio_49_51": line_noise_ratio,

        "notch_applied": APPLY_NOTCH,
        "notch_frequency_hz": NOTCH_FREQ if APPLY_NOTCH else np.nan,
        "notch_q": NOTCH_Q if APPLY_NOTCH else np.nan,

        "welch_nperseg": WELCH_NPERSEG,
        "welch_noverlap": WELCH_NOVERLAP,
        "welch_nfft": WELCH_NFFT,
        "frequency_lowcut_hz": LOWCUT,
        "frequency_highcut_hz": HIGHCUT,
    }

    for band_name, (low, high) in POWER_BANDS.items():
        power = compute_band_power(freqs_band, psd_band, low, high)

        features[f"power_{band_name}"] = power

        if total_power > 0 and np.isfinite(total_power):
            features[f"relative_power_{band_name}"] = power / total_power
        else:
            features[f"relative_power_{band_name}"] = np.nan

    return features


def add_pre_ratios(features_long):
    features_long = features_long.copy()

    ratio_features = [
        "mnf_10cycles",
        "mdf_10cycles",
        "total_power_10cycles",
        "power_20_60",
        "power_60_90",
        "power_90_150",
        "power_150_250",
        "power_250_450",
        "relative_power_20_60",
        "relative_power_60_90",
        "relative_power_90_150",
        "relative_power_150_250",
        "relative_power_250_450",
    ]

    for feature in ratio_features:
        if feature in features_long.columns:
            features_long[f"{feature}_ratio"] = np.nan
            features_long[f"log_{feature}_ratio"] = np.nan

    for (_, _, _), group in features_long.groupby(
        ["subject_id", "session", "muscle"],
        dropna=False,
    ):
        baseline = group[
            (group["trial_number"] == 1)
            & (group["feature_status"] == "ok")
        ]

        if baseline.empty:
            continue

        baseline_row = baseline.iloc[0]
        group_idx = group.index

        for feature in ratio_features:
            if feature not in features_long.columns:
                continue

            baseline_value = baseline_row[feature]

            if (
                pd.notna(baseline_value)
                and np.isfinite(baseline_value)
                and baseline_value > 0
            ):
                ratio = features_long.loc[group_idx, feature] / baseline_value

                features_long.loc[group_idx, f"{feature}_ratio"] = ratio

                valid_idx = ratio.index[
                    ratio.notna()
                    & np.isfinite(ratio)
                    & (ratio > 0)
                ]

                features_long.loc[
                    valid_idx,
                    f"log_{feature}_ratio"
                ] = np.log(ratio.loc[valid_idx])

    return features_long


def make_missing_row(base_info, muscle, selected_column, reason):
    row = base_info.copy()

    row.update(
        {
            "muscle": muscle,
            "selected_column": selected_column if selected_column is not None else "",
            "mnf_10cycles": np.nan,
            "mdf_10cycles": np.nan,
            "peak_frequency_10cycles": np.nan,
            "peak_power_10cycles": np.nan,
            "total_power_10cycles": np.nan,
            "total_power_sum_10cycles": np.nan,

            "peak_frequency_no50_10cycles": np.nan,
            "peak_power_no50_10cycles": np.nan,
            "line_noise_power_49_51": np.nan,
            "line_noise_ratio_49_51": np.nan,

            "power_20_60": np.nan,
            "power_60_90": np.nan,
            "power_90_150": np.nan,
            "power_150_250": np.nan,
            "power_250_450": np.nan,

            "relative_power_20_60": np.nan,
            "relative_power_60_90": np.nan,
            "relative_power_90_150": np.nan,
            "relative_power_150_250": np.nan,
            "relative_power_250_450": np.nan,

            "notch_applied": APPLY_NOTCH,
            "notch_frequency_hz": NOTCH_FREQ if APPLY_NOTCH else np.nan,
            "notch_q": NOTCH_Q if APPLY_NOTCH else np.nan,

            "welch_nperseg": np.nan,
            "welch_noverlap": np.nan,
            "welch_nfft": np.nan,
            "frequency_lowcut_hz": LOWCUT,
            "frequency_highcut_hz": HIGHCUT,
            "feature_status": reason,
        }
    )

    return row



def main():
    if not INVENTORY_FILE.exists():
        raise FileNotFoundError(f"Inventory file not found: {INVENTORY_FILE}")

    inventory = pd.read_excel(INVENTORY_FILE, sheet_name="inventory")
    inventory.columns = [str(column).strip() for column in inventory.columns]

    data_files = inventory[
        (inventory["include"].apply(normalize_text) == "si")
        & (inventory["status"].apply(normalize_text) == "ok")
        & (inventory["read_status"].apply(normalize_text) == "ok")
    ].copy()

    data_files["trial_number"] = pd.to_numeric(
        data_files["trial_number"],
        errors="coerce",
    )

    data_files = data_files.sort_values(
        by=["subject_id", "name_folder", "session", "trial_number"]
    ).copy()

    print(f"Files to be processed: {len(data_files)}")

    feature_rows = []

    for _, file_row in data_files.iterrows():
        csv_path = Path(file_row["file_path"])

        base_info = {
            "subject_id": file_row["subject_id"],
            "name_folder": file_row["name_folder"],
            "session": file_row["session"],
            "trial_number": int(file_row["trial_number"]),
            "trial_cat": trial_number_to_trial_cat(file_row["trial_number"]),
            "expected_side": normalize_text(file_row["expected_side"]),
            "shared": normalize_text(file_row["shared"]),
            "swap_muscles": normalize_text(file_row["swap_muscles"]),
            "file_name": file_row["file_name"],
            "file_path": str(csv_path),
        }

        try:
            df = read_emg_csv(csv_path)
        except Exception as exc:
            for muscle in ["biceps", "triceps"]:
                feature_rows.append(
                    make_missing_row(
                        base_info=base_info,
                        muscle=muscle,
                        selected_column=None,
                        reason=f"csv_read_error: {exc}",
                    )
                )
            continue

        columns = list(df.columns)

        biceps_column = find_muscle_column(
            columns,
            "biceps",
            base_info["expected_side"],
        )

        triceps_column = find_muscle_column(
            columns,
            "triceps",
            base_info["expected_side"],
        )

        if base_info["swap_muscles"] == "si":
            actual_columns = {
                "biceps": triceps_column,
                "triceps": biceps_column,
            }
        else:
            actual_columns = {
                "biceps": biceps_column,
                "triceps": triceps_column,
            }

        try:
            selection_info = select_central_window(
                n_samples=len(df),
                fs=FS,
            )

            base_info.update(selection_info)

        except Exception as exc:
            base_info.update(
                {
                    "cycle_selection_method": "central_fixed_window_from_metronome_protocol",
                    "cycle_selection_status": f"selection_error: {exc}",
                    "expected_cycle_duration_s": protocol_cycle_duration_sec(),
                    "expected_cycle_frequency_hz": 1.0 / protocol_cycle_duration_sec(),
                    "n_cycles_selected": N_CENTRAL_CYCLES,
                    "segment_duration_s": N_CENTRAL_CYCLES * protocol_cycle_duration_sec(),
                    "segment_start_sample": np.nan,
                    "segment_end_sample": np.nan,
                    "central_window_samples": CENTRAL_WINDOW_SAMPLES,
                    "segment_start_s": np.nan,
                    "segment_end_s": np.nan,
                    "recording_duration_s": len(df) / FS,
                }
            )

            for muscle in ["biceps", "triceps"]:
                feature_rows.append(
                    make_missing_row(
                        base_info=base_info,
                        muscle=muscle,
                        selected_column=actual_columns.get(muscle),
                        reason=f"cycle_selection_error: {exc}",
                    )
                )
            continue

        for muscle in ["biceps", "triceps"]:
            selected_column = actual_columns[muscle]

            if selected_column is None:
                feature_rows.append(
                    make_missing_row(
                        base_info=base_info,
                        muscle=muscle,
                        selected_column=selected_column,
                        reason="missing_muscle_column",
                    )
                )
                continue

            try:
                features = compute_frequency_features(
                    signal=df[selected_column],
                    segment_start_sample=selection_info["segment_start_sample"],
                    segment_end_sample=selection_info["segment_end_sample"],
                    fs=FS,
                )

                row = base_info.copy()
                row.update(
                    {
                        "muscle": muscle,
                        "selected_column": selected_column,
                        **features,
                        "feature_status": "ok",
                    }
                )

            except Exception as exc:
                row = make_missing_row(
                    base_info=base_info,
                    muscle=muscle,
                    selected_column=selected_column,
                    reason=f"frequency_feature_error: {exc}",
                )

            feature_rows.append(row)

    features_long = pd.DataFrame(feature_rows)
    features_long = add_pre_ratios(features_long)

    clean_columns = [
        "subject_id",
        "name_folder",
        "session",
        "trial_number",
        "trial_cat",
        "muscle",
        "file_name",
        "mnf_10cycles",
        "mnf_10cycles_ratio",
        "log_mnf_10cycles_ratio",
        "feature_status",
        "peak_frequency_10cycles",
    ]

    clean_columns = [col for col in clean_columns if col in features_long.columns]
    results_clean = features_long[clean_columns].copy()

    valid_mnf_mask = (
        (results_clean["feature_status"] == "ok")
        & results_clean["mnf_10cycles_ratio"].notna()
        & np.isfinite(results_clean["mnf_10cycles_ratio"])
        & (results_clean["mnf_10cycles_ratio"] > 0)
        & results_clean["log_mnf_10cycles_ratio"].notna()
        & np.isfinite(results_clean["log_mnf_10cycles_ratio"])
    )

    results_clean["include_analysis"] = valid_mnf_mask
    results_clean["qc_flag"] = "ok"

    results_clean.loc[
        results_clean["feature_status"] != "ok",
        "qc_flag",
    ] = "frequency_feature_problem"

    invalid_mnf_mask = (results_clean["feature_status"] == "ok") & ~valid_mnf_mask
    results_clean.loc[invalid_mnf_mask, "qc_flag"] = "invalid_mnf_ratio"

    line_noise_peak_mask = (
        results_clean["peak_frequency_10cycles"].notna()
        & np.isfinite(results_clean["peak_frequency_10cycles"])
        & (results_clean["peak_frequency_10cycles"] >= LINE_NOISE_LOW)
        & (results_clean["peak_frequency_10cycles"] <= LINE_NOISE_HIGH)
    )

    results_clean.loc[
        line_noise_peak_mask & (results_clean["qc_flag"] == "ok"),
        "qc_flag",
    ] = "possible_50hz_line_noise_peak_review"

    columns_for_clean_excel = [
        "subject_id",
        "name_folder",
        "session",
        "trial_number",
        "trial_cat",
        "muscle",
        "file_name",
        "mnf_10cycles",
        "mnf_10cycles_ratio",
        "log_mnf_10cycles_ratio",
        "feature_status",
        "include_analysis",
        "qc_flag",
    ]
    columns_for_clean_excel = [col for col in columns_for_clean_excel if col in results_clean.columns]
    results_clean_export = results_clean[columns_for_clean_excel].copy()


    issues = results_clean[
        results_clean["feature_status"] != "ok"
    ].copy()

    summary_status = (
        results_clean
        .groupby(
            ["subject_id", "name_folder", "session", "muscle"],
            dropna=False,
        )
        .agg(
            n_trials=("trial_number", "count"),
            ok_trials=(
                "feature_status",
                lambda values: sum(value == "ok" for value in values),
            ),
            feature_statuses=(
                "feature_status",
                lambda values: " | ".join(sorted(set(map(str, values)))),
            ),
            qc_flags=(
                "qc_flag",
                lambda values: " | ".join(sorted(set(map(str, values)))),
            ),
        )
        .reset_index()
    )

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        results_clean_export.to_excel(writer, sheet_name="frequency_results", index=False)

    if SAVE_DEBUG_WORKBOOK:
        with pd.ExcelWriter(DEBUG_OUTPUT_FILE, engine="openpyxl") as writer:
            results_clean.to_excel(writer, sheet_name="frequency_results_clean", index=False)
            features_long.to_excel(writer, sheet_name="frequency_features_debug", index=False)
            issues.to_excel(writer, sheet_name="issues", index=False)
            summary_status.to_excel(writer, sheet_name="summary_status", index=False)

    print("Frequency feature extraction finished.")
    print(f"Clean file saved in:\n{OUTPUT_FILE}")
    if SAVE_DEBUG_WORKBOOK:
        print(f"Debug file saved in:\n{DEBUG_OUTPUT_FILE}")

    print("\nFrequency-feature extraction problems:")
    if issues.empty:
        print("None")
    else:
        print(
            issues[
                [
                    "subject_id",
                    "name_folder",
                    "session",
                    "trial_number",
                    "muscle",
                    "feature_status",
                    "file_name",
                ]
            ]
        )


if __name__ == "__main__":
    main()
