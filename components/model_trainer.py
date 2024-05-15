import os
import sys
from logging import Logger
from dataclasses import dataclass

from logger.mylogger import get_logger
from utils import CustomException, save_object, evaluate_models, get_path
from logger.customize_logging import MyJSONFormatter, NonErrorFilter


from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


class ModelTrainer:
    def __init__(self, logger: Logger):
        self.trained_model_file_path=os.path.join("artifacts","model.pkl")
        self._logger = logger

    def initiate_model_trainer(self, data, preprocessor):
        try:
            self._logger.info("Getting features and target")
            features = data.drop(columns=['Price_euros'],axis=1)
            self._logger.info(f"features shape {features.shape}")
            target =  data['Price_euros']
            self._logger.info(f"target shape {target.shape}")
            
            models = {
                "RandomForest": RandomForestRegressor(random_state=1),
                "DecisionTree": DecisionTreeRegressor(random_state=1),
                "GradientBoosting": GradientBoostingRegressor(random_state=1),
                "LinearRegression": LinearRegression(),
                "RidgeRegression": Ridge(),
                "LassoRegression": Lasso(),
                "XGBRegressor": XGBRegressor(random_state=1),
                "K-NeighborsRegressor":  KNeighborsRegressor()
            }

            params= {
                "DecisionTree": {
                    'model__criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    "model__min_samples_leaf": [1, 3, 5]
                },
                "RandomForest":{
                    'model__criterion':['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                    'model__bootstrap': [True],
                    'model__max_features':['sqrt','log2',None],
                    'model__n_estimators': [32,64,128,256]
                },
                "GradientBoosting":{
                    'model__loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'model__learning_rate':[.1,.01,.05,.001],
                    'model__subsample':[0.6,0.7,0.8,0.9],
                    'model__criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'model__n_estimators': [16,32,64,128,256]
                },
                "LinearRegression":{},
                "RidgeRegression":{"model__alpha": [1.0, 2.0, 3.0]},
                "LassoRegression":{"model__alpha": [1.0, 2.0, 3.0]},
                "XGBRegressor":{
                    'model__learning_rate':[.1,.01,.05,.001],
                    'model__n_estimators': [16,32,64,128,256],
                    "model__max_depth": [3, 6, 9, 12],
                    "model__objective": ["reg:squaredlogerror", "reg:squarederror", "reg:pseudohubererror"]

                },
                "K-NeighborsRegressor": {
                    "model__n_neighbors": [10, 15, 20, 25, 30],
                    "model__weights": ['uniform', 'distance']
                }
                
            }

            self._logger.info("Staring evaluating models")
            model_report:list=evaluate_models(features=features, target=target, preprocessor=preprocessor,
                                             models=models,params=params)
            
            sorted_report = sorted(model_report, key=lambda x: x['best_score'], reverse=True)
            self._logger.info("Best model is found after evaluation")
            best_model_report = sorted_report[0]

            if best_model_report ['best_score']<0.7:
                self._logger.warn(f"Best found model only has score lower than 0.7")
            
            self._logger.info(f"Best model is {best_model_report['model_name']} with params {best_model_report['best_param']} and score {best_model_report['best_score']}")

            save_object(
                file_path=self.trained_model_file_path,
                obj=best_model_report['best_model']
            )
                
        except Exception as e:
            self._logger.exception(f"Error while training")
            raise CustomException(e,sys)