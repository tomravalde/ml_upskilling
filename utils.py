import pandas as pd
###################################################################################################
# Object to store datasets
###################################################################################################
"""
Import this object into other files for easy access to the data
"""

class DataFrameCollection:
    """
    # Example usage
    df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df2 = pd.DataFrame({'X': [7, 8, 9], 'Y': [10, 11, 12]})

    collection = DataFrameCollection()
    collection.add_dataframe('first_df', df1)
    collection.add_dataframe('second_df', df2)

    print(collection)
    # Output: DataFrameCollection(2 dataframes: first_df, second_df)

    # Retrieve a DataFrame
    retrieved_df = collection.get_dataframe('first_df')
    print(retrieved_df)
    # Output:
    #    A  B
    # 0  1  4
    # 1  2  5
    # 2  3  6

    # List all DataFrames in the collection
    print(collection.list_dataframes())
    # Output: ['first_df', 'second_df']

    # Remove a DataFrame
    collection.remove_dataframe('second_df')
    print(collection)
    # Output: DataFrameCollection(1 dataframe: first_df)
    """
    def __init__(self):
        # Initialize an empty dictionary to store DataFrames
        self.dataframes = {}

    def add_dataframe(self, name, dataframe):
        # Add a DataFrame to the collection
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("The provided object is not a pandas DataFrame.")
        self.dataframes[name] = dataframe

    def remove_dataframe(self, name):
        # Remove a DataFrame from the collection by its name
        if name in self.dataframes:
            del self.dataframes[name]
        else:
            raise KeyError(f"No DataFrame found with the name: {name}")

    def get_dataframe(self, name):
        # Retrieve a DataFrame by its name
        if name in self.dataframes:
            return self.dataframes[name]
        else:
            raise KeyError(f"No DataFrame found with the name: {name}")

    def list_dataframes(self):
        # List all DataFrames in the collection
        return list(self.dataframes.keys())

    def __repr__(self):
        # Provide a string representation of the collection
        return f"DataFrameCollection({len(self.dataframes)} dataframes: {', '.join(self.dataframes.keys())})"



