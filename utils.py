import os
import sys
import inspect
import pickle

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def error_message_detail(error: str,error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occured in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message[{error}]"

    return error_message

class CustomException(Exception):
    def __init__(self,error_message: str,error_detail:sys):
        super().__init__()
        self.error_message=error_message_detail(error=error_message, error_detail=error_detail)
    
    def __str__(self):
        return self.error_message

def get_path(module_path: str, cls) -> str:

    current_script_dir = os.path.dirname(module_path)
    cls_path = inspect.getfile(cls)
    path_parts = os.path.relpath(cls_path, current_script_dir).split('/')
    path_to_class = '.'.join(path_parts[1:])
    
    return path_to_class.replace('py', cls.__name__)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(features, target, preprocessor, models, params):
    model_report = []

    for model_name in models:
        model = models.get(model_name)
        param = params.get(model_name)

        model_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        gs = RandomizedSearchCV(estimator=model_pipeline, param_distributions=param, cv=3, scoring='r2',n_iter=5, return_train_score=False)
        gs.fit(features, target)

        model_report.append({
            "model_name": model_name,
            "best_score": gs.best_score_,
            "best_param": gs.best_params_,
            "best_model": gs.best_estimator_
        })

    return model_report

    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)