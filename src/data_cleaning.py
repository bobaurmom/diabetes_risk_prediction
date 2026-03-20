import pandas as pd

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = file_path

    def clean_data(self):
        try:
            print(f"Reading {self.file_path}")
            df = pd.read_csv(self.file_path)
            # print(f"Number of duplicates: {df.duplicated().sum()}")
            # df = df.drop_duplicates()
            # Turning yes/no to 1/0
            df = df.replace({'Yes': 1, 'No': 0})

            # Turning male/female to 1/0
            df = df.replace({'Male': 1, 'Female': 0})

            # Turning pos/neg to 1/0
            df = df.replace({'Positive': 1, 'Negative': 0})
            print("Data cleaning complete.")
            return df
        except FileNotFoundError:
            print("Error: The dataset file was not found!")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
    
# clean the data
cleaner = DataCleaner('data/diabetes_risk_prediction_dataset.csv')
cleaned_df = cleaner.clean_data()

#Save it to a new file
if cleaned_df is not None:
    cleaned_df.to_csv('data/clean_diabetes2.csv', index=False) #index=False to avoid adding an extra index column
    print("Saved cleaned data")
