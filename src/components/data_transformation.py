import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self) -> None:
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self) -> ColumnTransformer:
        '''
        Creates and returns a ColumnTransformer for numerical and categorical data.
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Define pipelines
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
            ])

            # Create ColumnTransformer
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            return preprocessor

        except Exception as e:
            logging.error(f"Error in creating transformer object: {e}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str) -> tuple[np.ndarray, np.ndarray, str]:
        '''
        Reads data from CSV, applies transformations, and saves the preprocessor object.
        '''
        try:
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Data read from CSV completed.")

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "math_score"

            # Prepare training and testing data
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Data transformation in progress.")

            logging.info("Applying transformation")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Concatenate features and target")
            train_arr = np.c_[input_feature_train_arr, target_feature_train_df.to_numpy()]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df.to_numpy()]
            
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info(f"Preprocessing object saved at {self.config.preprocessor_obj_file_path}.")

            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise CustomException(e, sys)
