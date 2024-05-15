# Machine Learning Project (Laptop Price Prediction)
This project is an end-to-end implemntation of building a machine learning model to predict laptop price
## 📝 Table of Contents
- [Setup & Requirements](#-setup--requirements)
- [Usage](#-usage)


## 🛠 Setup & Requirements
1. **Python Libraries**: Install the required Python libraries:
   - pandas
   - scikit-learn
   - numpy
   - seaborn
   - matplotlib
   - xgboost


## 🚀 Usage
1. **Ínstall requirements**: Run command pip install -r `requirements.txt`.
2. **Logging**: `logger` directory holds modules for logging in project. `mylogger.py` containes get_logger function to create logger from the `logging_config.json` while `customize_logging.py` is used to create custom formatter and filter.
3. **Notebook**: Exploratory data analysis and cleaning of the data is in `eda.ipynb` and model traing is done in `train_model.ipynb`.
4. **Modular model pipeline**: in `components` directory, train_prepper.py contains data ingestion and creation preprocessor pipeline. `model_trainer.py` contains the model training pipeline.
5. **Utils**: `utils.py` contains functions that are useful for the project.