from pathlib import Path
import re
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_DIR / "data" / "raw"
OUTPUT_DIR = REPO_DIR / "outputs"

METADATA_FILE = REPO_DIR / "data" / "metadata_subjects.xlsx"
OUTPUT_INVENTORY = OUTPUT_DIR / "inventory_emg_files.xlsx"


OUTPUT_DIR.mkdir(exist_ok=True)


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def extract_trial_number(file_name):
    stem = Path(file_name).stem
    numbers = re.findall(r"\d+", stem)

    if not numbers:
        return None

    return int(numbers[-1])


def read_csv_basic_info(csv_path):
    try:
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, sep=";", encoding="latin1")
        except Exception as e:
            return {
                "read_status": "error",
                "read_error": str(e),
                "n_columns": None,
                "column_names": None,
                "n_samples": None,
            }
    except Exception as e:
        return {
            "read_status": "error",
            "read_error": str(e),
            "n_columns": None,
            "column_names": None,
            "n_samples": None,

        }


    return {
        "read_status": "ok",
        "read_error": "",
        "n_columns": df.shape[1],
        "column_names": " | ".join(df.columns.astype(str)),
        "n_samples": df.shape[0],
    }


def session_folder_name(session):
    if session == "control":
        return "CONTROL"
    if session == "vibration":
        return "VIBRACIÓN"
    raise ValueError(f"Session not recognised: {session}")


##carga de metadatos
if not METADATA_FILE.exists():
    raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")

metadata = pd.read_excel(METADATA_FILE)

metadata.columns = [str(col).strip() for col in metadata.columns]

required_columns = [
    "subject_id",
    "name_folder",
    "include",
    "age",
    "sex",
    "dominant_arm",
    "control_shared",
    "control_side",
    "control_swap_muscles",
    "vibration_shared",
    "vibration_side",
    "vibration_swap_muscles",
    "notes",
]

missing_columns = [col for col in required_columns if col not in metadata.columns]

if missing_columns:
    raise ValueError(
        "Missing columns in metadata_subjects.xlsx:\n"
        + "\n".join(missing_columns)
    )

##Create inventory
inventory_rows = []

for _, subject in metadata.iterrows():
    subject_id = subject["subject_id"]
    name_folder = subject["name_folder"]
    include = normalize_text(subject["include"])

    subject_folder = DATA_DIR / str(name_folder)

    for session in ["control", "vibration"]:
        expected_session_folder = session_folder_name(session)
        session_folder = subject_folder / expected_session_folder

        if session == "control":
            expected_side = normalize_text(subject["control_side"])
            shared = normalize_text(subject["control_shared"])
            swap_muscles = normalize_text(subject["control_swap_muscles"])
        else:
            expected_side = normalize_text(subject["vibration_side"])
            shared = normalize_text(subject["vibration_shared"])
            swap_muscles = normalize_text(subject["vibration_swap_muscles"])

        if not subject_folder.exists():
            inventory_rows.append({
                "subject_id": subject_id,
                "name_folder": name_folder,
                "include": include,
                "session": session,
                "expected_session_folder": expected_session_folder,
                "folder_exists": "no",
                "file_name": "",
                "file_path": "",
                "trial_number": None,
                "expected_side": expected_side,
                "shared": shared,
                "swap_muscles": swap_muscles,
                "read_status": "",
                "read_error": "",
                "n_columns": None,
                "column_names": "",
                "n_samples": None,
                "status": "subject_folder_not_found",
            })
            continue

        if not session_folder.exists():
            inventory_rows.append({
                "subject_id": subject_id,
                "name_folder": name_folder,
                "include": include,
                "session": session,
                "expected_session_folder": expected_session_folder,
                "folder_exists": "no",
                "file_name": "",
                "file_path": "",
                "trial_number": None,
                "expected_side": expected_side,
                "shared": shared,
                "swap_muscles": swap_muscles,
                "read_status": "",
                "read_error": "",
                "n_columns": None,
                "column_names": "",
                "n_samples": None,
                "status": "session_folder_not_found",
            })
            continue

        csv_files = sorted(session_folder.glob("*.csv"))

        if len(csv_files) == 0:
            inventory_rows.append({
                "subject_id": subject_id,
                "name_folder": name_folder,
                "include": include,
                "session": session,
                "expected_session_folder": expected_session_folder,
                "folder_exists": "yes",
                "file_name": "",
                "file_path": "",
                "trial_number": None,
                "expected_side": expected_side,
                "shared": shared,
                "swap_muscles": swap_muscles,
                "read_status": "",
                "read_error": "",
                "n_columns": None,
                "column_names": "",
                "n_samples": None,
                "status": "no_csv_files_found",
            })
            continue

        for csv_path in csv_files:
            trial_number = extract_trial_number(csv_path.name)
            csv_info = read_csv_basic_info(csv_path)

            status = "ok"
            if trial_number is None:
                status = "trial_number_not_detected"
            elif trial_number < 1 or trial_number > 8:
                status = "trial_number_out_of_expected_range"
            elif csv_info["read_status"] != "ok":
                status = "csv_read_error"

            inventory_rows.append({
                "subject_id": subject_id,
                "name_folder": name_folder,
                "include": include,
                "session": session,
                "expected_session_folder": expected_session_folder,
                "folder_exists": "yes",
                "file_name": csv_path.name,
                "file_path": str(csv_path),
                "trial_number": trial_number,
                "expected_side": expected_side,
                "shared": shared,
                "swap_muscles": swap_muscles,
                "read_status": csv_info["read_status"],
                "read_error": csv_info["read_error"],
                "n_columns": csv_info["n_columns"],
                "column_names": csv_info["column_names"],
                "n_samples": csv_info["n_samples"],
                "status": status,
            })


inventory = pd.DataFrame(inventory_rows)

##summary by subject and susseion
summary = (
    inventory
    .groupby(["subject_id", "name_folder", "include", "session"], dropna=False)
    .agg(
        n_csv_files=("file_name", lambda x: sum(str(v).lower().endswith(".csv") for v in x)),
        detected_trials=("trial_number", lambda x: sorted([int(v) for v in x.dropna().unique()])),
        min_samples=("n_samples", "min"),
        max_samples=("n_samples", "max"),
        statuses=("status", lambda x: " | ".join(sorted(set(map(str, x))))),
    )
    .reset_index()
)

def expected_trials_status(row):
    if row["include"] != "si":
        return "excluded_subject"

    expected = list(range(1, 9))
    detected = row["detected_trials"]

    if detected == expected:
        return "ok"

    missing = sorted(set(expected) - set(detected))
    extra = sorted(set(detected) - set(expected))

    messages = []
    if missing:
        messages.append(f"missing_trials_{missing}")
    if extra:
        messages.append(f"extra_trials_{extra}")

    return " | ".join(messages)

summary["trial_check"] = summary.apply(expected_trials_status, axis=1)

inventory_clean_columns = [
    "subject_id", "name_folder", "include", "session", "expected_session_folder",
    "file_name", "file_path", "trial_number", "expected_side", "shared",
    "swap_muscles", "read_status", "status", "n_samples",
]
inventory_clean_columns = [col for col in inventory_clean_columns if col in inventory.columns]
inventory_clean = inventory[inventory_clean_columns].copy()

summary_clean_columns = [
    "subject_id", "name_folder", "include", "session", "n_csv_files",
    "detected_trials", "trial_check", "statuses", "min_samples", "max_samples",
]
summary_clean_columns = [col for col in summary_clean_columns if col in summary.columns]
summary_clean = summary[summary_clean_columns].copy()

#save results
with pd.ExcelWriter(OUTPUT_INVENTORY, engine="openpyxl") as writer:
    inventory_clean.to_excel(writer, sheet_name="inventory", index=False)
    summary_clean.to_excel(writer, sheet_name="summary", index=False)

print("Invntory created successfully")
print(f"File saved in:\n{OUTPUT_INVENTORY}")

print("\nQuick summary:")
print(summary[["subject_id", "name_folder", "include", "session", "n_csv_files", "detected_trials", "trial_check"]])
