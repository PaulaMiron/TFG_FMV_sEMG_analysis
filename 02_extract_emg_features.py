from pathlib import Path
import unicodedata
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

REPO_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = REPO_DIR / "outputs"

INVENTORY_FILE = BASE_DIR / "inventory_emg_files.xlsx"

OUTPUT_FILE = BASE_DIR / "RMS_statistics_results_FULL.xlsx"
CLEAN_OUTPUT_FILE = BASE_DIR / "RMS_statistics_results.xlsx"

#Mdurance sampling frequency
FS = 1024

#EMG filtering parameters
LOWCUT = 20
HIGHCUT = 450
FILTER_ORDER = 4

# RMS envelope parameters
RMS_WINDOW_SEC = 0.25
MOVEMENT_START_SEC = 5
BASELINE_SEARCH_SECONDS = 5
BASELINE_STABLE_SECONDS = 4


#Protocol parameters
METRONOME_BPM = 60
DIRECTION_CHANGES_PER_CYCLE = 2
N_CENTRAL_CYCLES = 10
CYCLE_DURATION_SECONDS = DIRECTION_CHANGES_PER_CYCLE * 60 / METRONOME_BPM
CENTRAL_WINDOW_SECONDS = N_CENTRAL_CYCLES * CYCLE_DURATION_SECONDS


CENTRAL_WINDOW_FEATURES = [
    "rms_total_10cycles",
    "rms_mean_10cycles",
]


def normalize_text(value):
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))

    return text



def read_emg_csv(csv_path):
    try:
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8", decimal=",")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, sep=";", encoding="latin1", decimal=",")

    df.columns = [str(c).strip() for c in df.columns]

    #convert all columns to numeric
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",",".", regex=False)
                .str.strip()
            )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df

def make_missing_row(base_info, muscle, reason):
    row = base_info.copy()
    row.update({
        "muscle": muscle,
        "selected_column": "",
        "n_total_samples": 0,
        "n_valid_samples": 0,
        "nan_ratio": np.nan,
        "raw_mean": np.nan,
        "raw_std": np.nan,
        "raw_min": np.nan,
        "raw_max": np.nan,
        "raw_range": np.nan,
        "rms_raw": np.nan,
        "rms_filtered": np.nan,
        "rms_baseline_points_used": np.nan,
        "rms_total_10cycles": np.nan,
        "rms_mean_10cycles": np.nan,     
        "feature_status": reason,
        "baseline_method": "",
        "baseline_source_trial": "",
        "baseline_start_sample": np.nan,
        "baseline_end_sample": np.nan,
        "baseline_mean": np.nan,
        "baseline_sd": np.nan,
        "baseline_threshold": np.nan,
        "baseline_qc_flag": "",
    })
    return row



def find_muscle_column ( columns, muscle, side):
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
            or f"_{side}_" in col_norm
            or f" {side} " in col_norm
            or col_norm.endswith(side)
            or side in col_norm
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


##EMG FILTERING
def interpolate_nans(x):
    x = pd.Series(x, dtype="float64")

    if x.isna().all():
        return x.to_numpy()

    x = x.interpolate(limit_direction="both")
    return x.to_numpy(dtype=float)


def filter_emg_signal(signal, fs=FS, lowcut=LOWCUT, highcut=HIGHCUT, order=FILTER_ORDER):
    # Applies a 4th order buterworth band-pass filter to the raw emg signal.
    
    x = pd.to_numeric(pd.Series(signal), errors="coerce").to_numpy(dtype=float)
    x = interpolate_nans(x)

    if np.isnan(x).all():
        raise ValueError("Signal contains only NaN values")
    if len(x) < 3 * fs:
        raise ValueError("signal too short for reliable filtering")
    
    nyquist = fs/2
    low = lowcut / nyquist
    high = highcut / nyquist

    if high >= 1:
        raise ValueError("Highcut frequency must be lower than Nyquist frequency")

    sos = butter(order, [low,high], btype="bandpass", output="sos")
    x_filtered = sosfiltfilt(sos,x)
    x_filtered = x_filtered - np.mean(x_filtered)

    return x_filtered


def compute_rms_envelope_4hz(x_filtered, fs=FS, window_sec=RMS_WINDOW_SEC):
    # Computes a RMS envelope using non-overlapping windows of 250ms, 4 Hz

    x_filtered = np.asarray(x_filtered, dtype=float)
    window_samples = int(round(window_sec * fs))

    if window_samples <= 0:
        raise ValueError("Window size must be greater than 0")

    n_windows = len(x_filtered) // window_samples
    if n_windows == 0:
        raise ValueError("Signal too short for reliable RMS envelope")
    
    x_cut = x_filtered[:n_windows * window_samples]
    x_windows = x_cut.reshape(n_windows, window_samples)
    rms_envelope = np.sqrt(np.mean(x_windows**2, axis=1))

    t_envelope = (
        np.arange(n_windows) * window_samples + window_samples / 2
    ) / fs
    
    return rms_envelope, t_envelope

def baseline_correct_rms_envelope(rms_envelope, t_envelope, fs=FS, search_seconds=BASELINE_SEARCH_SECONDS, stable_seconds=BASELINE_STABLE_SECONDS):
    rms_envelope = np.asarray(rms_envelope, dtype=float)
    t_envelope = np.asarray(t_envelope, dtype=float)

    search_idx = np.where(t_envelope < search_seconds)[0]

    stable_points = int(round(stable_seconds / RMS_WINDOW_SEC))

    if len(search_idx) < stable_points:
        baseline_info = {
            "baseline_method": "stable_4s_initial_rest",
            "baseline_source_trial": "same_trial",
            "baseline_start_sample": np.nan,
            "baseline_end_sample": np.nan,
            "baseline_mean": np.nan,
            "baseline_sd": np.nan,
            "baseline_level": np.nan,
            "baseline_threshold": np.nan,
            "baseline_qc_flag": "not_enough_baseline_points",
            "rms_baseline_points_used": len(search_idx),
        }

        corrected_envelope = np.full_like(rms_envelope, np.nan)

        return corrected_envelope, baseline_info

    best_score = np.inf
    best_idx = None

    for start_position in range(0, len(search_idx) - stable_points + 1):
        candidate_idx = search_idx[start_position:start_position + stable_points]
        candidate_values = rms_envelope[candidate_idx]

        candidate_mean = float(np.mean(candidate_values))
        candidate_sd = float(np.std(candidate_values, ddof=1))
        candidate_threshold = candidate_mean + 2 * candidate_sd

        candidate_score = candidate_threshold

        if candidate_score < best_score:
            best_score = candidate_score
            best_idx = candidate_idx

    baseline_values = rms_envelope[best_idx]

    baseline_mean = float(np.mean(baseline_values))
    baseline_sd = float(np.std(baseline_values, ddof=1))
    
    baseline_threshold = baseline_mean + 2 * baseline_sd
    # Level used for subtraction.
    baseline_level = baseline_threshold   

    baseline_start_s = max(
        0,
        t_envelope[best_idx[0]] - RMS_WINDOW_SEC / 2,
    )
    baseline_end_s = baseline_start_s + stable_seconds

    baseline_start_sample = int(round(baseline_start_s * fs))
    baseline_end_sample = int(round(baseline_end_s * fs))

    corrected_envelope = rms_envelope - baseline_level
    corrected_envelope[corrected_envelope < 0] = 0

    baseline_info = {
        "baseline_method": "stable_4s_initial_rest",
        "baseline_source_trial": "same_trial",
        "baseline_start_sample": baseline_start_sample,
        "baseline_end_sample": baseline_end_sample,
        "baseline_mean": baseline_mean,
        "baseline_sd": baseline_sd,
        "baseline_level": baseline_level,
        "baseline_threshold": baseline_threshold,
        "baseline_qc_flag": "ok",
        "rms_baseline_points_used": len(best_idx),
    }

    return corrected_envelope, baseline_info


def protocol_cycle_duration_sec(metronome_bpm=METRONOME_BPM, direction_changes_per_cycle=DIRECTION_CHANGES_PER_CYCLE):
    if metronome_bpm <= 0:
        raise ValueError("Metronome BPM must be greater than zero")
    return direction_changes_per_cycle * 60 / metronome_bpm


def select_central_metronome_window(n_samples, fs=FS, movement_start_sec=MOVEMENT_START_SEC, n_cycles=N_CENTRAL_CYCLES, metronome_bpm=METRONOME_BPM):
    # 60bmp --> flexion-extension = 2sec. 10 cycles --> 20sec
    total_duration_sec = n_samples/fs
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
    segment_start_sample = max(movement_start_sample, min(segment_start_sample, movement_end_sample - segment_samples))
    segment_end_sample = segment_start_sample + segment_samples

    return {
        "cycle_selection_method": "central_fixed_window_from_metronome_protocol",
        "cycle_selection_status": "ok",
        "expected_cycle_duration_s": cycle_duration_sec,
        "expected_cycle_frequency_hz": 1.0/cycle_duration_sec,
        "n_cycles_selected": n_cycles,
        "segment_duration_s": segment_duration_sec,
        "segment_start_sample": segment_start_sample,
        "segment_end_sample": segment_end_sample,
        "central_window_samples": segment_samples,
        "movement_start_sample": movement_start_sample,
        "movement_end_sample": movement_end_sample,
        "segment_start_s": segment_start_sample / fs,
        "segment_end_s": segment_end_sample / fs,
        "recording_duration_s": total_duration_sec,
        "movement_duration_s": total_duration_sec - movement_start_sec,
    }

def compute_emg_features(signal, segment_start_sample, segment_end_sample, fs=FS):
    x_raw_all = pd.to_numeric(pd.Series(signal), errors="coerce").to_numpy(dtype=float)

    n_total = len(x_raw_all)
    n_nan = np.isnan(x_raw_all).sum()
    nan_ratio = float(n_nan / n_total) if n_total > 0 else np.nan

    x_raw_valid = x_raw_all[~np.isnan(x_raw_all)]

    if len(x_raw_valid) == 0:
       raise ValueError("Signal contains only NaN values")

    #raw variables for quality control
    raw_mean = float(np.mean(x_raw_valid))
    raw_std = float(np.std(x_raw_valid, ddof=1)) if len(x_raw_valid) > 1 else np.nan
    raw_min = float(np.min(x_raw_valid))
    raw_max = float(np.max(x_raw_valid))
    raw_range = float(np.max(x_raw_valid) - np.min(x_raw_valid))
    rms_raw = float(np.sqrt(np.mean(x_raw_valid**2)))

    # main features from filtered emg
    x_filtered = filter_emg_signal(x_raw_all, fs=fs)
    rms_filtered = float(np.sqrt(np.mean(x_filtered**2)))

    rms_envelope, t_envelope = compute_rms_envelope_4hz(
        x_filtered,
        fs=fs,
    )

    rms_envelope_corrected, baseline_info = baseline_correct_rms_envelope(
        rms_envelope=rms_envelope,
        t_envelope=t_envelope,
        fs=fs,
        search_seconds=BASELINE_SEARCH_SECONDS,
        stable_seconds=BASELINE_STABLE_SECONDS,
    )

    x_segment = x_filtered[segment_start_sample:segment_end_sample]

    if len(x_segment) == 0:
        raise ValueError("Selected central window contains no EMG samples")

    segment_start_s = segment_start_sample / fs
    segment_end_s = segment_end_sample / fs

    env_mask = (
        (t_envelope >= segment_start_s)
        & (t_envelope < segment_end_s)
    )
    env_segment = rms_envelope_corrected[env_mask]

    if len(env_segment) == 0:
        raise ValueError("Selected central window contains no RMS-envelope samples")

    return {
        "n_total_samples": n_total,
        "n_valid_samples": len(x_raw_valid),
        "nan_ratio": nan_ratio,
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "raw_range": raw_range,
        "rms_raw": rms_raw,
        "rms_filtered": rms_filtered,

        "rms_baseline_points_used": baseline_info["rms_baseline_points_used"],

        "baseline_method": baseline_info["baseline_method"],
        "baseline_source_trial": baseline_info["baseline_source_trial"],
        "baseline_start_sample": baseline_info["baseline_start_sample"],
        "baseline_end_sample": baseline_info["baseline_end_sample"],
        "baseline_mean": baseline_info["baseline_mean"],
        "baseline_sd": baseline_info["baseline_sd"],
        "baseline_level": baseline_info["baseline_level"],
        "baseline_threshold": baseline_info["baseline_threshold"],
        "baseline_qc_flag": baseline_info["baseline_qc_flag"],

        "rms_total_10cycles": float(np.sum(env_segment)),
        "rms_mean_10cycles": float(np.mean(env_segment)),
        "filter_lowcut_hz": LOWCUT,
        "filter_highcut_hz": HIGHCUT,
        "filter_order": FILTER_ORDER,
    }

def add_baseline_ratios(features_long):
    # trial 1 is pre value
    features_long = features_long.copy()

    feature_aliases = {
        "rms_filtered": "rms_ratio",
    }

    for source_feature, ratio_column in feature_aliases.items():
        features_long[ratio_column] = np.nan
    
    for feature in CENTRAL_WINDOW_FEATURES:
        features_long[f"{feature}_ratio"] = np.nan

    for(_, _, _), group in features_long.groupby(
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

        for source_feature, ratio_column in feature_aliases.items():
            baseline_value = baseline_row[source_feature]

            if pd.notna(baseline_value) and baseline_value != 0:
                features_long.loc[group_idx, ratio_column] = (
                    features_long.loc[group_idx, source_feature]
                    / baseline_value
                )
        
        for feature in CENTRAL_WINDOW_FEATURES:
            baseline_value = baseline_row[feature]

            if pd.notna(baseline_value) and baseline_value != 0:
                features_long.loc[group_idx, f"{feature}_ratio"] = (
                    features_long.loc[group_idx, feature]
                    / baseline_value
                )

    return features_long
    

def build_wide_table(features_long):
    id_columns = [
        "subject_id",
        "name_folder",
        "session",
        "trial_number",
        "expected_side",
        "shared",
        "swap_muscles",
        "file_name",
        "file_path",
    ]

    values = [
        "rms_filtered",
        "rms_ratio",
    ]

    for feature in CENTRAL_WINDOW_FEATURES:
        values.extend([feature, f"{feature}_ratio"])

    wide = features_long.pivot_table(
        index=id_columns,
        columns="muscle",
        values=values,
        aggfunc="first",
    )

    wide.columns = [
        f"{variable}_{muscle}"
        for variable, muscle in wide.columns
    ]
    wide = wide.reset_index()

    trial_qc_columns = [
        "subject_id",
        "name_folder",
        "session",
        "trial_number",
        "file_name",
        "cycle_selection_method",
        "cycle_selection_status",
        "expected_cycle_duration_s",
        "expected_cycle_frequency_hz",
        "n_cycles_selected",
        "segment_duration_s",
        "segment_start_s",
        "segment_end_s",
        "recording_duration_s",
        "movement_duration_s",
    ]

    trial_qc = (
        features_long[trial_qc_columns]
        .drop_duplicates(
            subset=[
                "subject_id",
                "session",
                "trial_number",
                "file_name",
            ]
        )
    )

    wide = wide.merge(
        trial_qc,
        on=[
            "subject_id",
            "name_folder",
            "session",
            "trial_number",
            "file_name",
        ],
        how="left",
    )

    return wide, trial_qc

def check_periodicity_10cycles(df, biceps_column, triceps_column, selection_info, subject_id, session, trial_number, fs=FS, expected_frequency_hz=None, tolerance_hz=0.10):

    def get_dominant_frequency(rms_segment, rms_fs):
        x = np.asarray(rms_segment, dtype=float)
        x = x[np.isfinite(x)]

        if len(x) < 8:
            return np.nan

        x = x - np.mean(x)

        if np.std(x) == 0:
            return np.nan

        freqs = np.fft.rfftfreq(len(x), d=1 / rms_fs)
        spectrum = np.abs(np.fft.rfft(x))

        # Ignore DC and focus on plausible movement frequencies
        valid_band = (freqs >= 0.20) & (freqs <= 1.20)

        if not np.any(valid_band):
            return np.nan

        freqs_band = freqs[valid_band]
        spectrum_band = spectrum[valid_band]

        if np.max(spectrum_band) == 0:
            return np.nan

        return float(freqs_band[np.argmax(spectrum_band)])

    def scale_signal(x):
        x = np.asarray(x, dtype=float)
        x = x - np.nanmean(x)
        sd = np.nanstd(x)

        if sd == 0 or np.isnan(sd):
            return x

        return x / sd

    try:
        if biceps_column is None or triceps_column is None:
            print(
                f"{subject_id} | {session} | T{trial_number} | REVIEW | "
                f"missing biceps/triceps column"
            )
            return {"periodicity_status": "REVIEW"}

        expected_frequency_hz = (
            expected_frequency_hz
            if expected_frequency_hz is not None
            else selection_info["expected_cycle_frequency_hz"]
        )

        segment_start_s = selection_info["segment_start_sample"] / fs
        segment_end_s = selection_info["segment_end_sample"] / fs
        segment_duration_s = segment_end_s - segment_start_s

        # Filter signals 
        biceps_filtered = filter_emg_signal(df[biceps_column], fs=fs)
        triceps_filtered = filter_emg_signal(df[triceps_column], fs=fs)

        # RMS envelope at 4 Hz
        biceps_rms, t_biceps = compute_rms_envelope_4hz(biceps_filtered, fs=fs)
        triceps_rms, t_triceps = compute_rms_envelope_4hz(triceps_filtered, fs=fs)

        # Keep only the RMS-envelope samples corresponding to the selected central window
        biceps_mask = (t_biceps >= segment_start_s) & (t_biceps < segment_end_s)
        triceps_mask = (t_triceps >= segment_start_s) & (t_triceps < segment_end_s)

        biceps_rms_window = biceps_rms[biceps_mask]
        triceps_rms_window = triceps_rms[triceps_mask]

        min_len = min(len(biceps_rms_window), len(triceps_rms_window))

        if min_len < 8:
            print(
                f"{subject_id} | {session} | T{trial_number} | REVIEW | "
                f"not enough RMS-envelope samples in central window"
            )
            return {"periodicity_status": "REVIEW"}

        biceps_rms_window = biceps_rms_window[:min_len]
        triceps_rms_window = triceps_rms_window[:min_len]

        biceps_scaled = scale_signal(biceps_rms_window)
        triceps_scaled = scale_signal(triceps_rms_window)
        alternating_signal = biceps_scaled - triceps_scaled

        rms_fs = 1 / RMS_WINDOW_SEC

        biceps_freq = get_dominant_frequency(biceps_rms_window, rms_fs)
        triceps_freq = get_dominant_frequency(triceps_rms_window, rms_fs)
        alternance_freq = get_dominant_frequency(alternating_signal, rms_fs)

        biceps_ok = (
            np.isfinite(biceps_freq)
            and abs(biceps_freq - expected_frequency_hz) <= tolerance_hz
        )
        triceps_ok = (
            np.isfinite(triceps_freq)
            and abs(triceps_freq - expected_frequency_hz) <= tolerance_hz
        )
        half_cycle_frequency_hz = 2 * expected_frequency_hz

        alternance_ok = (
            np.isfinite(alternance_freq)
            and (
                abs(alternance_freq - expected_frequency_hz) <= tolerance_hz
                or abs(alternance_freq - half_cycle_frequency_hz) <= tolerance_hz
            )
        )
        n_ok = sum([biceps_ok, triceps_ok, alternance_ok])

        if n_ok == 3:
            status = "OK"
        elif n_ok == 2:
            status = "PARTIAL_OK"
        else:
            status = "REVIEW"

        cycle_freqs = [
            freq for freq in [biceps_freq, triceps_freq, alternance_freq]
            if (
                np.isfinite(freq)
                and abs(freq - expected_frequency_hz) <= tolerance_hz
            )
        ]

        estimated_cycles = (
            float(np.mean(cycle_freqs) * segment_duration_s)
            if len(cycle_freqs) > 0
            else np.nan
        )

        print(
            f"{subject_id} | {session} | T{trial_number} | {status} | "
            f"biceps={biceps_freq:.2f} Hz | "
            f"triceps={triceps_freq:.2f} Hz | "
            f"alternance={alternance_freq:.2f} Hz | "
            f"estimated_cycles={estimated_cycles:.1f}"
        )

        return {
            "periodicity_status": status,
            "biceps_frequency_hz": biceps_freq,
            "triceps_frequency_hz": triceps_freq,
            "alternance_frequency_hz": alternance_freq,
            "estimated_cycles": estimated_cycles,
        }

    except Exception as exc:
        print(
            f"{subject_id} | {session} | T{trial_number} | REVIEW | "
            f"periodicity_error: {exc}"
        )
        return {"periodicity_status": "REVIEW"}

def main():
    if not INVENTORY_FILE.exists():
        raise FileNotFoundError(f"Inventory file not found: {INVENTORY_FILE}")

    inventory = pd.read_excel(INVENTORY_FILE, sheet_name="inventory")
    inventory.columns = [
        str(column).strip()
        for column in inventory.columns
    ]

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
    periodicity_results = []

    for _, file_row in data_files.iterrows():
        csv_path = Path(file_row["file_path"])

        base_info = {
            "subject_id": file_row["subject_id"],
            "name_folder": file_row["name_folder"],
            "session": file_row["session"],
            "trial_number": int(file_row["trial_number"]),
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
                        base_info,
                        muscle,
                        f"csv_read_error: {exc}",
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
            selection_info = select_central_metronome_window(
                n_samples=len(df),
                fs=FS,
            )
            base_info.update(selection_info)

            periodicity_result = check_periodicity_10cycles(
                df=df,
                biceps_column=actual_columns["biceps"],
                triceps_column=actual_columns["triceps"],
                selection_info=selection_info,
                subject_id=base_info["subject_id"],
                session=base_info["session"],
                trial_number=base_info["trial_number"],
                fs=FS,
            )
            periodicity_results.append(periodicity_result)

        except Exception as exc:
            base_info.update(
                {
                    "cycle_selection_method": "central_fixed_window_from_metronome_protocol",
                    "cycle_selection_status": f"selection_error: {exc}",
                    "expected_cycle_duration_s": protocol_cycle_duration_sec(),
                    "expected_cycle_frequency_hz": 1.0 / protocol_cycle_duration_sec(),
                    "n_cycles_selected": N_CENTRAL_CYCLES,
                    "segment_duration_s": N_CENTRAL_CYCLES * protocol_cycle_duration_sec(),
                    "segment_start_s": np.nan,
                    "segment_end_s": np.nan,
                    "recording_duration_s": len(df) / FS,
                    "movement_duration_s": len(df) / FS - MOVEMENT_START_SEC,
                }
            )

            for muscle in ["biceps", "triceps"]:
                feature_rows.append(
                    make_missing_row(
                        base_info,
                        muscle,
                        f"cycle_selection_error: {exc}",
                    )
                )
            continue
    

        for muscle in ["biceps", "triceps"]:
            selected_column = actual_columns[muscle]

            if selected_column is None:
                feature_rows.append(
                    make_missing_row(
                        base_info,
                        muscle,
                        "missing_muscle_column",
                    )
                )
                continue

            try:
                features = compute_emg_features(
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
                    base_info,
                    muscle,
                    f"feature_extraction_error: {exc}",
                )
                row["selected_column"] = selected_column

            feature_rows.append(row)

    features_long = pd.DataFrame(feature_rows)
    features_long = add_baseline_ratios(features_long)

    clean_columns = [
        "subject_id",
        "name_folder",
        "session",
        "trial_number",
        "muscle",
        "file_name",
        "rms_mean_10cycles",
        "rms_mean_10cycles_ratio",
        "feature_status",
        "baseline_qc_flag",
    ]

    clean_columns = [col for col in clean_columns if col in features_long.columns]
    results_clean = features_long[clean_columns].copy()

    results_clean["include_analysis"] = (
        (results_clean["feature_status"] == "ok")
        & results_clean["rms_mean_10cycles_ratio"].notna()
        & np.isfinite(results_clean["rms_mean_10cycles_ratio"])
        & (results_clean["rms_mean_10cycles_ratio"] > 0)
    )

    results_clean["qc_flag"] = "ok"

    feature_problem_mask = results_clean["feature_status"] != "ok"
    results_clean.loc[feature_problem_mask, "qc_flag"] = "feature_extraction_problem"

    missing_ratio_mask = (
        (results_clean["feature_status"] == "ok")
        & results_clean["rms_mean_10cycles_ratio"].isna()
    )
    results_clean.loc[missing_ratio_mask, "qc_flag"] = "missing_rms_ratio"

    invalid_ratio_mask = (
        results_clean["rms_mean_10cycles_ratio"].notna()
        & (
            ~np.isfinite(results_clean["rms_mean_10cycles_ratio"])
            | (results_clean["rms_mean_10cycles_ratio"] <= 0)
        )
    )
    results_clean.loc[invalid_ratio_mask, "qc_flag"] = "invalid_rms_ratio"

    baseline_debug_columns = [
        "subject_id",
        "name_folder",
        "session",
        "trial_number",
        "muscle",
        "file_name",
        "baseline_method",
        "baseline_source_trial",
        "baseline_start_sample",
        "baseline_end_sample",
        "baseline_mean",
        "baseline_sd",
        "baseline_threshold",
        "baseline_level",  
        "baseline_qc_flag",
        "rms_mean_10cycles",
        "rms_total_10cycles",
        "feature_status",
    ]

    baseline_debug_columns = [
        col for col in baseline_debug_columns
        if col in features_long.columns
    ]

    baseline_debug = features_long[baseline_debug_columns].copy()

    wide, trial_qc = build_wide_table(features_long)

    issues = features_long[
        features_long["feature_status"] != "ok"
    ].copy()


    summary_status = (
        features_long
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
        )
        .reset_index()
    )

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        features_long.to_excel(
            writer,
            sheet_name="features_long",
            index=False,
        )
        wide.to_excel(
            writer,
            sheet_name="features_wide",
            index=False,
        )
        trial_qc.to_excel(
            writer,
            sheet_name="trial_qc",
            index=False,
        )
        issues.to_excel(
            writer,
            sheet_name="issues",
            index=False,
        )
        summary_status.to_excel(
            writer,
            sheet_name="summary_status",
            index=False,
        )
        baseline_debug.to_excel(
            writer,
            sheet_name="baseline_debug",
            index=False,
        )

    with pd.ExcelWriter(CLEAN_OUTPUT_FILE, engine="openpyxl") as writer:
        results_clean.to_excel(
            writer,
            sheet_name="results_clean",
            index=False,
        )

    print("Feature extraction finished.")
    print(f"File saved in:\n{OUTPUT_FILE}")
    print(f"Clean results file saved in:\n{CLEAN_OUTPUT_FILE}")

    print("\nFeature-extraction problems:")
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

    print("\nPeriodicity summary:")

    if len(periodicity_results) == 0:
        print("No periodicity checks were performed.")
    else:
        total_periodicity_trials = len(periodicity_results)

        for status in ["OK", "PARTIAL_OK", "REVIEW"]:
            count = sum(
                result.get("periodicity_status") == status
                for result in periodicity_results
            )
            percentage = 100 * count / total_periodicity_trials

            print(
                f"{status}: {count}/{total_periodicity_trials} trials "
                f"({percentage:.1f}%)"
            )


if __name__ == "__main__":
    main()
