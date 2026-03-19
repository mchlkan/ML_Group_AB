"""Paths, constants, and feature lists loaded from training_summary.json."""

import json
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    if start is None:
        start = Path(__file__).resolve().parent.parent
    for candidate in [start, *start.parents]:
        if (candidate / "datasets").exists() and (candidate / "requirements.txt").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate project root from {start}. "
        "Expected a parent containing 'datasets/' and 'requirements.txt'."
    )


ROOT = find_project_root()

# --- Directories ---
DATA_DIR = ROOT / "datasets" / "v3_features"
MODEL_DIR = ROOT / "artifacts" / "models" / "xgboost_final_pipeline"
EVAL_DIR = ROOT / "artifacts" / "evaluations" / "xgboost_final_pipeline"
V2_DATA_DIR = ROOT / "datasets" / "v2" / "full"
COUNTRIES_CSV = ROOT / "datasets" / "Countries Data By Aadarsh Vani.csv"
CULTURAL_DIST_CSV = ROOT / "datasets" / "cultural_distance_matrix.csv"

# --- Splits ---
TRAIN_PATH = DATA_DIR / "train.parquet"
VAL_PATH = DATA_DIR / "val.parquet"
TEST_PATH = DATA_DIR / "test.parquet"

# --- Constants ---
RANDOM_STATE = 42
TOP_K = 5

# --- Load training summary ---
_SUMMARY_PATH = MODEL_DIR / "training_summary.json"
with open(_SUMMARY_PATH) as _f:
    TRAINING_SUMMARY = json.load(_f)

PRUNED_ROW_FEATURE_COLS: list[str] = TRAINING_SUMMARY["pruned_row_feature_cols"]
TRACK_FEATURE_COLS: list[str] = TRAINING_SUMMARY["track_feature_cols"]
FILL_VALUES_TRAIN: dict[str, float] = TRAINING_SUMMARY["fill_values_train"]
FILL_VALUES_FINAL: dict[str, float] = TRAINING_SUMMARY["fill_values_final"]

# --- Country list (62 Spotify markets) ---
COUNTRY_LIST = [
    "Andorra", "Argentina", "Australia", "Austria", "Belgium", "Bolivia",
    "Brazil", "Bulgaria", "Canada", "Chile", "Colombia", "Costa Rica",
    "Czech Republic", "Denmark", "Dominican Republic", "Ecuador",
    "El Salvador", "Estonia", "Finland", "France", "Germany", "Greece",
    "Guatemala", "Honduras", "Hong Kong", "Hungary", "Iceland", "India",
    "Indonesia", "Ireland", "Israel", "Italy", "Japan", "Latvia",
    "Lithuania", "Malaysia", "Mexico", "Netherlands", "New Zealand",
    "Nicaragua", "Norway", "Panama", "Paraguay", "Peru", "Philippines",
    "Poland", "Portugal", "Romania", "Singapore", "Slovakia",
    "South Africa", "Spain", "Sweden", "Switzerland", "Taiwan",
    "Thailand", "Turkey", "United Arab Emirates", "United Kingdom",
    "United States", "Uruguay", "Vietnam",
]

# --- Country → primary language mapping ---
COUNTRY_PRIMARY_LANG: dict[str, str] = {
    "Andorra": "ca", "Argentina": "es", "Australia": "en", "Austria": "de",
    "Belgium": "nl", "Bolivia": "es", "Brazil": "pt", "Bulgaria": "bg",
    "Canada": "en", "Chile": "es", "Colombia": "es", "Costa Rica": "es",
    "Czech Republic": "cs", "Denmark": "da", "Dominican Republic": "es",
    "Ecuador": "es", "El Salvador": "es", "Estonia": "et", "Finland": "fi",
    "France": "fr", "Germany": "de", "Greece": "el", "Guatemala": "es",
    "Honduras": "es", "Hong Kong": "zh", "Hungary": "hu", "Iceland": "is",
    "India": "hi", "Indonesia": "id", "Ireland": "en", "Israel": "he",
    "Italy": "it", "Japan": "ja", "Latvia": "lv", "Lithuania": "lt",
    "Malaysia": "ms", "Mexico": "es", "Netherlands": "nl",
    "New Zealand": "en", "Nicaragua": "es", "Norway": "no", "Panama": "es",
    "Paraguay": "es", "Peru": "es", "Philippines": "tl", "Poland": "pl",
    "Portugal": "pt", "Romania": "ro", "Singapore": "en", "Slovakia": "sk",
    "South Africa": "en", "Spain": "es", "Sweden": "sv",
    "Switzerland": "de", "Taiwan": "zh", "Thailand": "th", "Turkey": "tr",
    "United Arab Emirates": "ar", "United Kingdom": "en",
    "United States": "en", "Uruguay": "es", "Vietnam": "vi",
}

# --- Country → continent mapping ---
COUNTRY_CONTINENT: dict[str, str] = {
    "Andorra": "Europe", "Argentina": "South America", "Australia": "Oceania",
    "Austria": "Europe", "Belgium": "Europe", "Bolivia": "South America",
    "Brazil": "South America", "Bulgaria": "Europe", "Canada": "North America",
    "Chile": "South America", "Colombia": "South America",
    "Costa Rica": "North America", "Czech Republic": "Europe",
    "Denmark": "Europe", "Dominican Republic": "North America",
    "Ecuador": "South America", "El Salvador": "North America",
    "Estonia": "Europe", "Finland": "Europe", "France": "Europe",
    "Germany": "Europe", "Greece": "Europe", "Guatemala": "North America",
    "Honduras": "North America", "Hong Kong": "Asia", "Hungary": "Europe",
    "Iceland": "Europe", "India": "Asia", "Indonesia": "Asia",
    "Ireland": "Europe", "Israel": "Asia", "Italy": "Europe",
    "Japan": "Asia", "Latvia": "Europe", "Lithuania": "Europe",
    "Malaysia": "Asia", "Mexico": "North America", "Netherlands": "Europe",
    "New Zealand": "Oceania", "Nicaragua": "North America",
    "Norway": "Europe", "Panama": "North America",
    "Paraguay": "South America", "Peru": "South America",
    "Philippines": "Asia", "Poland": "Europe", "Portugal": "Europe",
    "Romania": "Europe", "Singapore": "Asia", "Slovakia": "Europe",
    "South Africa": "Africa", "Spain": "Europe", "Sweden": "Europe",
    "Switzerland": "Europe", "Taiwan": "Asia", "Thailand": "Asia",
    "Turkey": "Europe", "United Arab Emirates": "Asia",
    "United Kingdom": "Europe", "United States": "North America",
    "Uruguay": "South America", "Vietnam": "Asia",
}

# Map country name to the rank column key used in features
def country_to_rank_col(country: str) -> str:
    return "rank_" + country.lower().replace(" ", "_")

# --- Available song languages ---
SONG_LANGUAGES = [
    ("en", "English"), ("es", "Spanish"), ("pt", "Portuguese"),
    ("fr", "French"), ("de", "German"), ("it", "Italian"),
    ("nl", "Dutch"), ("ja", "Japanese"), ("ko", "Korean"),
    ("zh", "Chinese"), ("hi", "Hindi"), ("ar", "Arabic"),
    ("tr", "Turkish"), ("pl", "Polish"), ("sv", "Swedish"),
    ("ro", "Romanian"), ("id", "Indonesian"), ("th", "Thai"),
    ("tl", "Tagalog"), ("vi", "Vietnamese"), ("other", "Other"),
]

# --- Continent one-hot column names ---
CONTINENT_ONEHOT_COLS = [
    "target_continent_africa",
    "target_continent_asia",
    "target_continent_europe",
    "target_continent_north_america",
    "target_continent_oceania",
    "target_continent_south_america",
]
