import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered into DataIngestion Method:")
        try:
            
            df = pd.read_csv('data/StudentsPerformance.csv')
            logging.info("Read the dataset into a dataframe.")

            # To ensure the artifacts directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved to %s", self.ingestion_config.raw_data_path)

            logging.info("Train and test dataset splitting initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data splitting into train and test sets completed")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except FileNotFoundError as fnf_error:
            logging.error("File not found: %s", fnf_error)
            raise CustomException(fnf_error, sys)
        except pd.errors.EmptyDataError as ede_error:
            logging.error("No data: %s", ede_error)
            raise CustomException(ede_error, sys)
        except Exception as e:
            logging.error("An error occurred: %s", e)
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)