import pandas as pd

# === functions ===
def pivot_wider(df: pd.DataFrame, 
                id_cols: list[str] | str, 
                names_from: str, 
                values_from: str
               ) -> pd.DataFrame:
    """
    Reshapes a dataframe from long format to wide format using specified ID columns,
    column names, and values.
    
    Returns a dataframe in wide format with one row per unique ID combination.
    Arguments:
        df: input dataframe in long format
        id_cols: column name or list of column names that uniquely identify each row
        names_from: column whose unique values will become new column names
        values_from: column whose values will fill the new wide-format columns

    Returns:
        A dataframe reshaped to wide format.
    """
    if isinstance(id_cols, str):
        id_cols = [id_cols]
    
    df2 = df.groupby(id_cols + [names_from], as_index=False)[values_from].first()    
    value_wide = df2.pivot(
        index=id_cols,
        columns=names_from,
        values=values_from
    ).reset_index()

    df_merged = df.merge(value_wide, on=id_cols, how="left")

    df_wide = (
        df_merged
            .drop(columns=[
                names_from,
                values_from
            ])
            .drop_duplicates()
    )
    print("Data reshaped to wide format successfully.")
    return df_wide

