import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


@dataclass
class Config:
    data_dir: Path = Path("data/joai-competition-2025")
    train_csv: Path = data_dir / "train.csv"
    test_csv: Path = data_dir / "test.csv"
    images_dir: Path = data_dir / "images"
    output_dir: Path = Path(".")
    seed: int = 42
    noise_std_dev: float = 0.01

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(self.seed)
        print(f"Configuration loaded. Output directory: {self.output_dir}")

class FeatureEngineer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        try:
            self.train_df = pd.read_csv(cfg.train_csv)
            self.test_df = pd.read_csv(cfg.test_csv)
            print(f"Train data loaded: {cfg.train_csv}. Shape: {self.train_df.shape}")
            print(f"Test data loaded: {cfg.test_csv}. Shape: {self.test_df.shape}")
        except FileNotFoundError as e:
            print(f"Error loading CSV file: {e}")
            print(f"Please ensure {cfg.train_csv.name} and {cfg.test_csv.name} are in {cfg.data_dir}")
            raise

        # Ensure 'id' column exists; create one based on DataFrame index if missing
        if 'id' not in self.train_df.columns:
            print("Warning: 'id' column not found in train_df. Creating one from index.")
            self.train_df.insert(0, 'id', range(len(self.train_df)))
        if 'id' not in self.test_df.columns:
            print("Warning: 'id' column not found in test_df. Creating one from index.")
            self.test_df.insert(0, 'id', range(len(self.test_df)))

        self.sensors = sorted([
            col for col in self.train_df.columns if col.startswith("MQ")
        ])
        if not self.sensors:
            print("Warning: No sensor columns (starting with 'MQ') detected in train_df.")
        else:
            print(f"Detected sensors: {self.sensors}")

    def compute_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes basic text features like caption length (word count)."""
        if "Caption" in df.columns:
            df["caption_len"] = df["Caption"].fillna("").astype(str).apply(lambda x: len(x.split()))
        else:
            print("Warning: 'Caption' column not found, setting caption_len to 0.")
            df["caption_len"] = 0
        return df

_temp_pattern = re.compile(
    r'(?P<raw>(?:~?[+-]?\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?|[<>]\s?[+-]?\d+(?:\.\d+)?))\s?°?(?P<unit>[CFcf])'
)
_loc_pattern = re.compile(
    r'(?P<lat>\d+°\d+\'\d+(?:\.\d+)?\"[NS])\s*,?\s*(?P<lon>\d+°\d+\'\d+(?:\.\d+)?\"[EW])'
)
_colors_list = [
    'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink',
    'brown', 'gray', 'grey', 'black', 'white', 'cyan', 'magenta'
]
_color_pattern = re.compile(
    r'\b((?:' + '|'.join(_colors_list) + r')(?:-(?:' + '|'.join(_colors_list) + r'))?)\b',
    flags=re.IGNORECASE
)

def extract_temperatures(text: str) -> List[str]:
    """Extracts temperature strings (value + unit) from text."""
    if not isinstance(text, str): return []
    return [m.group('raw').strip() + m.group('unit').upper() for m in _temp_pattern.finditer(text)]

def extract_locations(text: str) -> List[str]:
    """Extracts location strings (lat, lon) from text."""
    if not isinstance(text, str): return []
    return [f"{m.group('lat')}, {m.group('lon')}" for m in _loc_pattern.finditer(text)]

def extract_colors(text: str) -> List[str]:
    """Extracts color names from text, converting to lowercase."""
    if not isinstance(text, str): return []
    return [m.group(1).lower() for m in _color_pattern.finditer(text)]

def extract_all_entities(text: Union[str, float]) -> Dict[str, List[str]]:
    """Extracts all defined entities (temperature, location, color) from text."""
    if pd.isna(text):
        text = ""
    else:
        text = str(text)

    return {
        "temperatures": extract_temperatures(text),
        "locations": extract_locations(text),
        "colors": extract_colors(text),
    }

def process_captions(df: pd.DataFrame, text_col: str = "Caption") -> pd.DataFrame:
    """Applies entity extraction to the text column of a DataFrame."""
    if text_col not in df.columns:
        print(f"Warning: Text column '{text_col}' not found. Skipping entity extraction.")
        for col in ['temperatures', 'locations', 'colors']:
            df[col] = [[] for _ in range(len(df))]
        return df

    entities = df[text_col].fillna("").apply(extract_all_entities)

    entities_df = pd.json_normalize(entities)

    for col in ['temperatures', 'locations', 'colors']:
        if col not in entities_df.columns:
            entities_df[col] = [[] for _ in range(len(entities_df))]

    df_reset = df.reset_index(drop=True)
    entities_df_reset = entities_df.reset_index(drop=True)

    return pd.concat([df_reset, entities_df_reset], axis=1)


def convert_to_celsius(value_str: Union[str, None], unit: str) -> float:
    """Converts a temperature value string (potentially with prefixes) to Celsius."""
    if value_str is None: return np.nan
    try:
        cleaned_value_str = value_str.lstrip('<>~').strip()
        if not cleaned_value_str: return np.nan
        fval = float(cleaned_value_str)
    except (ValueError, TypeError):
        return np.nan

    unit = unit.upper()
    if unit == 'C':
        return fval
    elif unit == 'F':
        return (fval - 32.0) * 5.0 / 9.0
    else:
        print(f"Warning: Unknown temperature unit '{unit}' encountered.")
        return np.nan

def add_temperature_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Parses extracted temperatures, converts to Celsius, and adds min/max bounds."""
    if 'temperatures' not in df.columns:
        print("Warning: 'temperatures' column not found. Skipping temperature bounds.")
        df['lower_temperature'] = np.nan
        df['higher_temperature'] = np.nan
        return df

    lows, highs = [], []
    for temp_list in df['temperatures']:
        if not isinstance(temp_list, list) or not temp_list:
            lows.append(np.nan); highs.append(np.nan); continue

        celsius_values = []
        for temp_str in temp_list:
            match = _temp_pattern.match(temp_str)
            if not match: continue # Skip if the string doesn't match the expected pattern

            raw_val, unit = match.group('raw').strip(), match.group('unit').upper()
            low_str, high_str = None, None # Initialize low/high string values

            if '-' in raw_val and not raw_val.startswith(('<', '>','~','+','-')):
                parts = raw_val.split('-', 1)
                if len(parts) == 2:
                    low_str, high_str = parts[0].strip(), parts[1].strip()
            elif raw_val.startswith(('<', '>')):
                low_str = high_str = raw_val[1:].strip()
            else:
                low_str = high_str = raw_val.lstrip('~').strip()

            low_c = convert_to_celsius(low_str, unit)
            high_c = convert_to_celsius(high_str, unit)

            if not np.isnan(low_c): celsius_values.append(low_c)
            if high_str != low_str and not np.isnan(high_c): celsius_values.append(high_c)

        lows.append(min(celsius_values) if celsius_values else np.nan)
        highs.append(max(celsius_values) if celsius_values else np.nan)

    df['lower_temperature'] = lows
    df['higher_temperature'] = highs
    return df

def encode_caption_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes extracted caption entities into numerical/categorical features
       and drops the original list-based columns."""

    if 'locations' in df.columns:
        df['has_location'] = df['locations'].apply(lambda x: 1 if isinstance(x, list) and len(x) > 0 else 0).astype(int) # Ensure integer type
        df.drop(columns=['locations'], inplace=True)
    else: df['has_location'] = 0

    if 'colors' in df.columns:
        df['num_colors'] = df['colors'].apply(lambda x: len(x) if isinstance(x, list) else 0).astype(int)
        df['first_color_name'] = df['colors'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'None').astype(str) # Ensure string type
        df.drop(columns=['colors'], inplace=True)
        top_colors = {'none', 'blue', 'green', 'red'}
        df['first_color_name'] = df['first_color_name'].apply(lambda x: x if x in top_colors else 'Other')
    else:
        df['num_colors'] = 0
        df['first_color_name'] = 'None'

    if 'temperatures' in df.columns:
        df.drop(columns=['temperatures'], inplace=True)

    return df


def main() -> None:
    """Main function to orchestrate the feature engineering pipeline."""
    cfg = Config()
    fe = FeatureEngineer(cfg)

    print("\nStarting feature engineering process...")

    print("[Step 1/10] Combining train and test data...")
    train_df_raw = fe.train_df.copy().assign(_is_train=1)
    test_df_raw = fe.test_df.copy().assign(_is_train=0)
    combined_df = pd.concat([train_df_raw, test_df_raw], ignore_index=True, sort=False)
    print(f"Combined data shape: {combined_df.shape}")
    if 'id' not in combined_df.columns:
        raise ValueError("'id' column lost during concatenation. Check input data.")

    print("[Step 2/10] Calculating sensor features...")
    sensors = fe.sensors
    numeric_feature_candidates = []

    if sensors:
        combined_df['sensor_mean_raw'] = combined_df[sensors].mean(axis=1)
        combined_df['sensor_std_raw'] = combined_df[sensors].std(axis=1)
        combined_df['sensor_max_raw'] = combined_df[sensors].max(axis=1)
        combined_df['sensor_min_raw'] = combined_df[sensors].min(axis=1)
        numeric_feature_candidates.extend(['sensor_mean_raw', 'sensor_std_raw', 'sensor_max_raw', 'sensor_min_raw'])

        for sensor in sensors:
            if sensor in combined_df.columns:
                mean_val = combined_df[sensor].mean() # Global mean for combined data
                combined_df[f"{sensor}_delta"] = combined_df[sensor] - mean_val
                numeric_feature_candidates.append(f"{sensor}_delta")

        for sensor in sensors:
            delta_col = f"{sensor}_delta"
            if delta_col in combined_df.columns:
                mean_delta = combined_df[delta_col].mean() # Should be near 0
                std_delta = combined_df[delta_col].std()
                norm_col = f"{sensor}_norm"
                combined_df[norm_col] = (combined_df[delta_col] - mean_delta) / (std_delta + 1e-6)
                combined_df[norm_col] = combined_df[norm_col].clip(-5, 5)
                numeric_feature_candidates.append(norm_col)

        if "MQ5_norm" in combined_df.columns and "MQ8_norm" in combined_df.columns:
            combined_df["MQ5_MQ8_norm_ratio"] = combined_df["MQ5_norm"] / (combined_df["MQ8_norm"] + 1e-6) # Add epsilon for stability
            combined_df["MQ5_MQ8_norm_diff"] = combined_df["MQ5_norm"] - combined_df["MQ8_norm"]
            numeric_feature_candidates.extend(["MQ5_MQ8_norm_ratio", "MQ5_MQ8_norm_diff"])

        smoke_norms = [f"{s}_norm" for s in ["MQ2", "MQ3", "MQ135"] if f"{s}_norm" in combined_df.columns]
        if smoke_norms:
            combined_df["smoke_index_norm"] = combined_df[smoke_norms].mean(axis=1)
            numeric_feature_candidates.append("smoke_index_norm")

        fuel_norms = [f"{s}_norm" for s in ["MQ5", "MQ6", "MQ8"] if f"{s}_norm" in combined_df.columns]
        if fuel_norms:
            combined_df["fuel_index_norm"] = combined_df[fuel_norms].mean(axis=1)
            numeric_feature_candidates.append("fuel_index_norm")

        if "MQ7_norm" in combined_df.columns:
            combined_df["co_index_norm"] = combined_df["MQ7_norm"]
            numeric_feature_candidates.append("co_index_norm")

        norm_cols = [col for col in combined_df.columns if col.endswith('_norm') and col in numeric_feature_candidates]
        if norm_cols:
            combined_df["sensor_norm_max"] = combined_df[norm_cols].max(axis=1)
            combined_df["sensor_norm_min"] = combined_df[norm_cols].min(axis=1)
            combined_df["sensor_norm_mean"] = combined_df[norm_cols].mean(axis=1)
            combined_df["sensor_norm_std"] = combined_df[norm_cols].std(axis=1)
            numeric_feature_candidates.extend(["sensor_norm_max", "sensor_norm_min", "sensor_norm_mean", "sensor_norm_std"])

        delta_cols = [f"{s}_delta" for s in sensors if f"{s}_delta" in combined_df.columns]
        if delta_cols:
            combined_df["max_abs_delta_sensor"] = combined_df[delta_cols].abs().idxmax(axis=1).str.replace("_delta", "")
    # Additional MQ8/MQ5 Raw Features
    if "MQ8" in combined_df.columns and "MQ5" in combined_df.columns:
        combined_df["mq8_mq5_raw_diff"] = combined_df["MQ8"] - combined_df["MQ5"]
        combined_df["mq8_mq5_raw_ratio"] = combined_df["MQ8"] / (combined_df["MQ5"] + 1e-6) # Epsilon for stability
        combined_df["mq8_mq5_raw_sum"] = combined_df["MQ8"] + combined_df["MQ5"]
        combined_df["mq8_sq"] = combined_df["MQ8"] ** 2
        combined_df["mq5_sq"] = combined_df["MQ5"] ** 2
        combined_df["mq_diff_abs"] = (combined_df["MQ8"] - combined_df["MQ5"]).abs()
        try:
            combined_df["mq8_bucket"] = pd.qcut(combined_df["MQ8"], q=5, labels=False, duplicates='drop') # Increased buckets
            combined_df["mq5_bucket"] = pd.qcut(combined_df["MQ5"], q=5, labels=False, duplicates='drop')
        except ValueError as e:
            print(f"Warning: Could not create buckets for MQ8/MQ5 (likely due to insufficient unique values): {e}")
            combined_df["mq8_bucket"] = 0 # Assign default bucket if qcut fails
            combined_df["mq5_bucket"] = 0

        numeric_feature_candidates.extend([
            "mq8_mq5_raw_diff", "mq8_mq5_raw_ratio", "mq8_mq5_raw_sum",
            "mq8_sq", "mq5_sq", "mq_diff_abs"
        ])

    print(f"Sensor features calculated. Shape: {combined_df.shape}")
    if 'id' not in combined_df.columns:
        raise ValueError("'id' column lost after sensor feature engineering.")

    print("[Step 3/10] Calculating basic text features (length)...")
    combined_df = fe.compute_text_features(combined_df)
    if "caption_len" in combined_df.columns:
        numeric_feature_candidates.append("caption_len") # Caption length is numeric
    print(f"Text features calculated. Shape: {combined_df.shape}")
    if 'id' not in combined_df.columns:
        raise ValueError("'id' column lost after text feature engineering.")

    print("[Step 4/10] Processing captions for entities (temperature, location, color)...")
    combined_processed_df = process_captions(combined_df, text_col="Caption")
    combined_processed_df = add_temperature_bounds(combined_processed_df)
    combined_processed_df = encode_caption_features(combined_processed_df) # Encodes and drops raw lists
    print(f"Caption entities processed. Shape: {combined_processed_df.shape}")
    if 'id' not in combined_processed_df.columns:
        raise ValueError("'id' column lost after caption processing pipeline.")

    if 'lower_temperature' in combined_processed_df.columns: numeric_feature_candidates.append('lower_temperature')
    if 'higher_temperature' in combined_processed_df.columns: numeric_feature_candidates.append('higher_temperature')
    if 'num_colors' in combined_processed_df.columns: numeric_feature_candidates.append('num_colors')

    print("[Step 5/10] One-hot encoding categorical features...")
    categorical_cols = []
    if 'max_abs_delta_sensor' in combined_processed_df.columns:
        categorical_cols.append('max_abs_delta_sensor')
    if 'first_color_name' in combined_processed_df.columns:
        categorical_cols.append('first_color_name')
    if 'mq8_bucket' in combined_processed_df.columns:
        categorical_cols.append('mq8_bucket')
    if 'mq5_bucket' in combined_processed_df.columns:
        categorical_cols.append('mq5_bucket')

    if categorical_cols:
        print(f"Columns to one-hot encode: {categorical_cols}")
        combined_final_df = pd.get_dummies(
            combined_processed_df,
            columns=categorical_cols,
            prefix=categorical_cols, # Use column name as prefix for clarity
            prefix_sep='_',
            dummy_na=False,
            dtype=int
        )
        print(f"Categorical features encoded. Shape: {combined_final_df.shape}")
        low_imp_cols = [
            'first_color_name_blue-green',
            'first_color_name_green-yellow',
            'first_color_name_orange',
            'first_color_name_red-yellow',
            'first_color_name_white',
            'mq5_bucket_1',
            'mq5_bucket_2'
        ]
        for col in low_imp_cols:
            if col in combined_final_df.columns:
                combined_final_df.drop(columns=[col], inplace=True)
    else:
        print("No categorical columns identified for one-hot encoding.")
        combined_final_df = combined_processed_df # Proceed without encoding

    if 'id' not in combined_final_df.columns:
        raise ValueError("'id' column lost after one-hot encoding.")

    print("[Step 6/10] Splitting back into train and test sets...")
    train_nn = combined_final_df[combined_final_df["_is_train"] == 1].copy()
    test_nn = combined_final_df[combined_final_df["_is_train"] == 0].copy()

    train_nn.drop(columns=["_is_train"], inplace=True)
    test_nn.drop(columns=["_is_train", "Gas"], inplace=True, errors='ignore')

    print(f"Train set shape after split: {train_nn.shape}")
    print(f"Test set shape after split: {test_nn.shape}")
    if 'id' not in train_nn.columns: raise KeyError("FATAL: 'id' column missing in train_nn after split.")
    if 'Gas' not in train_nn.columns: raise KeyError("FATAL: 'Gas' target column missing in train_nn after split.")
    if 'id' not in test_nn.columns: raise KeyError("FATAL: 'id' column missing in test_nn after split.")

    print("[Step 7/10] Separating IDs, labels, and features...")
    train_labels = train_nn['Gas']
    train_ids = train_nn['id']
    test_ids = test_nn['id']

    train_features = train_nn.drop(columns=['Gas', 'id',], errors='ignore')
    test_features = test_nn.drop(columns=['id', ], errors='ignore')
    print(f"Initial train features shape: {train_features.shape}")
    print(f"Initial test features shape: {test_features.shape}")

    print("[Step 8/10] Aligning columns between train and test feature sets...")
    train_features_aligned, test_features_aligned = train_features.align(
        test_features, join='inner', axis=1, copy=False # Modify in place if possible
    )
    aligned_cols = train_features_aligned.columns.tolist()
    print(f"Columns aligned. Train features shape: {train_features_aligned.shape}, Test features shape: {test_features_aligned.shape}")
    print(f"Number of aligned feature columns: {len(aligned_cols)}")
    if 'id' in aligned_cols:
        raise ValueError("'id' column ended up in aligned features. Check separation step.")

    print("[Step 9/10] Normalizing numeric features and adding noise...")

    cols_to_scale = []
    potential_numeric = train_features_aligned.select_dtypes(include=np.number).columns.tolist()

    ohe_prefixes = [f"{col}_" for col in categorical_cols] # e.g., 'first_color_name_'
    binary_flags = ['has_location'] # Add others if created

    for col in potential_numeric:
        is_one_hot = any(col.startswith(prefix) for prefix in ohe_prefixes)
        is_binary_flag = col in binary_flags

        if not is_one_hot and not is_binary_flag:
            if col in test_features_aligned.columns:
                 cols_to_scale.append(col)
            else:
                 print(f"Warning: Column {col} identified for scaling in train but missing in aligned test features. Skipping.")


    if cols_to_scale:
        print(f"Identified {len(cols_to_scale)} numeric columns for Robust Scaling.")
        scaler = RobustScaler()
        train_features_aligned[cols_to_scale] = scaler.fit_transform(train_features_aligned[cols_to_scale])
        test_features_aligned[cols_to_scale] = scaler.transform(test_features_aligned[cols_to_scale])
        print("RobustScaler applied to train and test sets.")

        if cfg.noise_std_dev > 0:
            print(f"Adding Gaussian noise (mean=0, std={cfg.noise_std_dev}) to scaled training features...")
            noise = np.random.normal(0, cfg.noise_std_dev, train_features_aligned[cols_to_scale].shape)
            train_features_aligned.loc[:, cols_to_scale] = train_features_aligned.loc[:, cols_to_scale].values + noise
            print("Noise added to training features.")
        else:
            print("Skipping noise addition (noise_std_dev set to 0).")

    else:
        print("No columns identified for scaling with RobustScaler.")


    print("[Step 10/10] Reconstructing final DataFrames and saving...")
    final_train_nn = pd.concat([
        train_ids.reset_index(drop=True),
        train_features_aligned.reset_index(drop=True),
        train_labels.reset_index(drop=True)
    ], axis=1)

    final_test_nn = pd.concat([
        test_ids.reset_index(drop=True),
        test_features_aligned.reset_index(drop=True)
    ], axis=1)

    final_test_cols = final_train_nn.drop(columns=['Gas']).columns.tolist()
    final_test_nn = final_test_nn[final_test_cols]

    train_output_path = cfg.output_dir / "train_nn_features.csv"
    test_output_path = cfg.output_dir / "test_nn_features.csv"

    final_train_nn.to_csv(train_output_path, index=False)
    final_test_nn.to_csv(test_output_path, index=False)

    print("-" * 60)
    print("Feature Engineering Pipeline Completed Successfully!")
    print(f"Processed Training Features:")
    print(f"  - Shape: {final_train_nn.shape}")
    print(f"  - Saved to: {train_output_path}")
    print(f"Processed Test Features:")
    print(f"  - Shape: {final_test_nn.shape}")
    print(f"  - Saved to: {test_output_path}")
    print("-" * 60)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print("Error: Input CSV file not found. Please check the `data_dir` path in the script.")
    except KeyError as e:
        print(f"Error: A required column is missing: {e}")
        print("This might happen after splitting or feature processing if columns were unexpectedly dropped.")
        import traceback
        traceback.print_exc()
    except ValueError as e:
        print(f"Error: A value-related issue occurred: {e}")
        print("This could be due to data inconsistencies or problems during processing like alignment.")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()