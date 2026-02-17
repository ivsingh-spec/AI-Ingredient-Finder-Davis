"""
merge_tables.py

Purpose:
Load data from the folder F1 Data - take the columns which are relevant and make a final dataset. 
Merge raw F1 CSV tables into a single race-driver dataset.
Each row represents one driver in one race. Of current drivers only
"""
"""
merge_tables.py

Purpose:
Load data from the folder F1 Data - take the columns which are relevant and make a final dataset. 
Merge raw F1 CSV tables into a single race-driver dataset.
Each row represents one driver in one race. Of current drivers only
"""
import pandas as pd

def merge_tables(dataframes):
    """
    Merges results, races, drivers, constructors, qualifying, and status tables.

    Args:
        dataframes (dict): Dictionary of pandas DataFrames loaded from CSVs

    Returns:
        pandas.DataFrame: Final merged race-driver dataset
    """

    results = dataframes["results"]
    races = dataframes["races"]
    drivers = dataframes["drivers"]
    constructors = dataframes["constructors"]
    qualifying = dataframes["qualifying"]
    status = dataframes["status"]

    # Select relevant columns


    results = results[
        [
            "raceId",
            "driverId",
            "constructorId",
            "grid",
            "positionOrder",
            "points",
            "laps",
            "milliseconds",
            "fastestLapSpeed",
            "statusId",
        ]
    ]

    races = races[
        [
            "raceId",
            "year",
            "round",
        ]
    ]

    drivers = drivers[
        [
            "driverId",
            "forename",
            "surname",
            "dob",
            "nationality",
        ]
    ]

    constructors = constructors[
        [
            "constructorId",
            "name",
            "nationality",
        ]
    ]

    qualifying = qualifying[
        [
            "raceId",
            "driverId",
            "position",
        ]
    ].rename(columns={"position": "quali_position"})

    status = status[
        [
            "statusId",
            "status",
        ]
    ]

    # Merge

    df = results.merge(races, on="raceId", how="left")
    df = df.merge(drivers, on="driverId", how="left")
    df = df.merge(constructors, on="constructorId", how="left")
    df = df.merge(qualifying, on=["raceId", "driverId"], how="left")
    df = df.merge(status, on="statusId", how="left")

    # Rename important columns

    df = df.rename(columns={
        "positionOrder": "finish_position",
        "forename": "driver_forename",
        "surname": "driver_surname",
        "name": "constructor_name",
        "nationality_x": "driver_nationality",
        "nationality_y": "constructor_nationality"
    })

    # Filter to modern era (after 2000)

    df = df[df["year"] >= 2000]

    # Convert numeric columns properly


    df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
    df["quali_position"] = pd.to_numeric(df["quali_position"], errors="coerce")

    # Drop rows with NA values

    df = df.dropna()

    # Clean column names

    df.columns = (
        df.columns
        .str.lower()
        .str.replace(" ", "_")
    )

    # Final sorting

    df = df.sort_values(["year", "round", "finish_position"])
    df = df.reset_index(drop=True)

    return df

