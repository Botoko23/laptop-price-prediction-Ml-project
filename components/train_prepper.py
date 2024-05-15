import os
import sys
from logging import Logger
from dataclasses import dataclass

from logger.mylogger import get_logger
from utils import get_path, CustomException, save_object
from components.model_trainer import ModelTrainer
from logger.customize_logging import MyJSONFormatter, NonErrorFilter

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
    

class TrainPrepper:
    def __init__(self, logger: Logger):
        self.raw_data_path: str=os.path.join('artifacts',"data.csv")
        self.preprocessor_obj_file_path: str=os.path.join('artifacts',"preprocessor.pkl")
        self._logger = logger

    def initiate_data_ingestion(self):
        self._logger.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook/data/cleaned_data.csv')
            self._logger.info('Simulate reading the dataset from source as dataframe')

            os.makedirs(os.path.dirname(self.raw_data_path),exist_ok=True)

            df.to_csv(self.raw_data_path,index=False,header=True)

            self._logger.info("Ingestion of the data is completed")

            return df
        
        except Exception as e:
            self._logger.exception('error while ingesting data')
            raise CustomException(str(e),sys)

    def get_data_transformer_object(self):
            '''
            This function is responsible for data transformation
            
            '''
            try:
                numerical_columns = ['Inches', 'Width', 'Height', 'isTouchscreen', 'CpuSpeed (GHz)',
                                    'MemorySize1', 'MemorySize2', 'Ram (GB)', 'Weight (kg)']
                categorical_columns = ['Company', 'TypeName', 'CpuProducer', 'MemoryType1', 'MemoryType2',
                                    'GpuProducer']

                num_pipeline= Pipeline(
                    steps=[
                    ("scaler",StandardScaler())

                    ]
                )

                cat_pipeline=Pipeline(

                    steps=[
                    ("one_hot_encoder",OneHotEncoder()),
                    ]

                )

                self._logger.info(f"Categorical columns: {categorical_columns}")
                self._logger.info(f"Numerical columns: {numerical_columns}")

                preprocessor=ColumnTransformer(
                    [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipelines",cat_pipeline,categorical_columns)

                    ]

                )

                save_object(

                    file_path=self.preprocessor_obj_file_path,
                    obj=preprocessor

                )

                self._logger.info('Preprocessor is created successfully')

                return preprocessor
            
            except Exception as e:
                self._logger.exception('Error while creating preprocessor')
                raise CustomException(e,sys)        
        
if __name__=="__main__":
    formatter_path = get_path(__file__, MyJSONFormatter)
    filter_path = get_path(__file__, NonErrorFilter)

    logger = get_logger('my_project', json_formatter_path=formatter_path, filter_path=filter_path)
    logger.info('logger is set up')
    
    logger.info('start processing')
    train_prepper=TrainPrepper(logger=logger)
    data =train_prepper.initiate_data_ingestion()
    preprocessor =train_prepper.get_data_transformer_object()

    logger.info('start training')
    modeltrainer=ModelTrainer(logger=logger)
    modeltrainer.initiate_model_trainer(data=data, preprocessor=preprocessor)
