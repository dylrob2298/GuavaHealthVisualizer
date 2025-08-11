import numpy as np
import pandas as pd

MOBILITY_AID_TYPES = {"Walker", "Manual Wheelchair", "Automatic Wheelchair"}


def load_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    return df


def prepare_symptoms_column(df: pd.DataFrame) -> pd.DataFrame:
    symptoms = (
        df[df["type"] == "Symptoms"]
        .groupby("datetime")["name"]
        .apply(lambda s: sorted({x for x in s.dropna()}))
        .rename("Symptoms")
    )
    return symptoms


def prepare_energy_column(df: pd.DataFrame) -> pd.Series:
    energy = (df[df["type"] == "Energy"]
              .dropna(subset=["datetime", "value"])
              .groupby("datetime")["value"].last()
              .rename("Energy")
            )
    return energy


def prepare_water_column(df: pd.DataFrame) -> pd.DataFrame:
    water_df = df[df["type"] == "Water"].copy()

    # Normalize dtypes to avoid incompatible partial assignment
    water_df["value"] = pd.to_numeric(water_df["value"], errors="coerce").astype("float64")
    water_df["unit"] = water_df["unit"].astype("string")

    # Convert any non-"mL" unit (assumed [foz_us]) to mL
    mask_not_ml = water_df["unit"].ne("mL") & water_df["unit"].notna()
    conv = 29.5735
    water_df["value_ml"] = np.where(mask_not_ml, water_df["value"] * conv, water_df["value"])
    water_df.loc[mask_not_ml, "unit"] = "mL"

    # Pivot to a single "Water" column in mL
    water = water_df.pivot(index="datetime", columns="type", values="value_ml")
    water = water.rename(columns={"Water": "Water"})  # keeps column name explicit if desired
    return water



def prepare_mobility_aid_column(df: pd.DataFrame) -> pd.DataFrame:
    mobility_aid = (
        df[df["type"].isin(MOBILITY_AID_TYPES)]
        .groupby("datetime")["type"]
        .apply(lambda s: sorted({x for x in s.dropna()}))
        .rename("Mobility Aids")
    )
    return mobility_aid


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    base_datetime_index = pd.Index(df["datetime"].dropna().unique(), name="datetime")
    base_datetime_index = pd.Index(sorted(base_datetime_index))
    out_df = pd.DataFrame(index=base_datetime_index)

    symptoms = prepare_symptoms_column(df)
    energy = prepare_energy_column(df)
    water = prepare_water_column(df)
    mobility_aid = prepare_mobility_aid_column(df)

    out_df = out_df.join(symptoms, how="left")
    out_df = out_df.join(energy, how="left")
    out_df = out_df.join(water, how="left")
    out_df = out_df.join(mobility_aid, how="left")

    return out_df


def main():
    print("Hello from guavahealthvisualizer!")
    df = load_csv("guava.csv")
    print(transform_data(df))

    df = transform_data(df)

    df.info()

    df.to_csv("transformed_guava.csv")


if __name__ == "__main__":
    main()
