import pandas as pd

# function to read the data file
def read_data(file_path, **kwargs):
    """
    Reads a CSV file from the given file path and returns a pandas DataFrame.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file.
    **kwargs : keyword arguments, optional
        Additional parameters to be passed to pandas.read_csv() like delimiter, encoding, etc.
    
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the data from the CSV file.
    """
    raw_data = pd.read_csv(file_path, **kwargs)
    return raw_data


# function to merge different data files
def merge_dataset(df1, df2, join_type, on_param):
    """
    Merges two DataFrames based on a common key and join type.
    
    Parameters:
    -----------
    df1 : pandas.DataFrame
        The first DataFrame.
    df2 : pandas.DataFrame
        The second DataFrame.
    join_type : str
        Type of merge to be performed. Options include 'left', 'right', 'inner', 'outer'.
    on_param : str or list
        Column(s) on which to merge the DataFrames.
    
    Returns:
    --------
    pandas.DataFrame
        A DataFrame resulting from the merge of df1 and df2.
    """
    final_df = df1.copy()
    final_df = final_df.merge(df2, how=join_type, on=on_param)
    return final_df


# function to drop columns from data
def drop_col(df, col_list):
    """
    Drops specified columns from a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame from which columns will be dropped.
    col_list : list
        List of column names to drop.
    
    Raises:
    -------
    ValueError
        If any column in col_list does not exist in the DataFrame.
    
    Returns:
    --------
    pandas.DataFrame
        The DataFrame with specified columns removed.
    """
    for col in col_list:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in dataframe.")
        else:
            df = df.drop(col, axis=1)
    return df


# function to remove null values
def null_values(df):
    """
    Removes rows with null values from a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame from which to remove null values.
    
    Returns:
    --------
    pandas.DataFrame
        The DataFrame with rows containing null values dropped.
    """
    df = df.dropna()
    return df


# function to find maximum value and returning maximum value and it's index
def max_val_index(l):
    """
    Finds the maximum value in a list and its index.
    
    Parameters:
    -----------
    l : list
        A list of numerical values.
    
    Returns:
    --------
    tuple
        A tuple containing the maximum value and its index (max_value, index).

    Example:
    --------
    max_value, index = max_val_index([1, 3, 5, 7])
    """
    max_l = max(l)
    max_index = l.index(max_l)
    return max_l, max_index