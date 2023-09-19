import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import logging
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from datasets import DatasetPipeline


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')


class MachineLearningPipeline(DatasetPipeline):
    # Directories Configurations
    root_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = f"{root_dir}/models"
    evaluations_dir = f"{root_dir}/evaluations"

    # Machine Learning Configurations
    n_estimators = 1000
    time_steps = 15
    model_filename = None

    # Script Configurations
    cultures = None
    culture_title = None
    culture_alias = None
    culture_id = None

    def start_pipeline(self, culture_alias, culture_id):
        self.create_models_directories()

        self.culture_alias = culture_alias
        self.culture_id = culture_id
        self.model_filename = f"{culture_alias}_{culture_id}_model.pkl"
        self.cultures = self.getter_datasets_json()
        self.culture_title = self.cultures[self.culture_alias][self.culture_id]["title"]

        logging.info("Starting Pipeline | {}".format(self.culture_title))

        df_raw = self.getter_dataset(
            culture_alias,
            culture_id,
            self.time_steps
        )
        df = self.dataset_config(df_raw)

        if self.model_filename not in os.listdir(self.models_dir):
            x, y = self.dataset_splitting(df)
            model = self.model_training(x, y)
        else:
            model = self.model_loading()

        predicted_values = self.model_prediction(
            model,
            df
        )

        logging.info("End Pipeline | {} -> {} predictions".format(self.culture_title, len(predicted_values)))


    def model_training(self, x, y):
        logging.info("Model training")

        model = XGBRegressor(n_estimators=self.n_estimators)

        self.model_evaluation(model, x, y)
        model.fit(x, y)

        return self.model_saving(model)


    def model_prediction(self, model, df):
        logging.info("Model prediction")

        df = df.drop(["date", "value"], axis=1)
        new_values = []

        for i in range(self.time_steps):
            df = self.dataset_predict_skip(df, new_values)

            predicted_value = float("{:.2f}".format(model.predict(df)[0]))
            new_values.append(predicted_value)
            logging.info("Predicted value - {}/{} -> {}".format(i+1, self.time_steps, predicted_value))

        return new_values


    def model_saving(self, model):
        logging.info("Model saving")

        with open(f"{self.models_dir}/{self.model_filename}", "wb") as fid:
            pickle.dump(model, fid)

        return model


    def model_loading(self):
        logging.info("Model loading")

        with open(f"{self.models_dir}/{self.model_filename}", "rb") as fid:
            model = pickle.load(fid)

        return model


    def model_evaluation(self, model, x, y, test_size=0.33):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        logging.info("Model score: {}".format(r2_score(y_pred, y_test, multioutput='variance_weighted')))

        predicted_data = pd.DataFrame({"y_pred":y_pred, "y_test": y_test}, index=x_test.index)
        predicted_data.reset_index(inplace=True)
        predicted_data.drop(["index"], axis=1, inplace=True)

        data_y_pred = predicted_data.loc[:,"y_pred"].copy()
        data_y_test = predicted_data.loc[:,"y_test"].copy()

        plt.figure(figsize=(30, 10))
        plt.plot(data_y_test, linestyle='dashed', color='b')
        plt.plot(data_y_pred, linestyle='solid', color='r')

        plt.legend(['Actual', "Predicted"], loc='best', prop={'size': 14})
        plt.title(f'Preco {self.culture_alias.title()}', weight='bold', fontsize=16)
        plt.ylabel('Real (R$)', weight='bold', fontsize=14)
        plt.xlabel('Dia', weight='bold', fontsize=14)
        plt.xticks(weight='bold', fontsize=12, rotation=45)
        plt.yticks(weight='bold', fontsize=12)
        plt.grid(color = 'y', linewidth='0.5')

        evaluation_dir = f"{self.evaluations_dir}/{self.culture_alias}_{self.culture_id}_eval.png"
        plt.savefig(evaluation_dir)

        logging.info("Model evaluation saved on : {}".format(evaluation_dir))

    def create_models_directories(self):
        logging.info("Creating models directories")

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        if not os.path.exists(self.evaluations_dir):
            os.makedirs(self.evaluations_dir)

if __name__ == "__main__":
    # Argument Parsing

    argParser = argparse.ArgumentParser()
    argParser.add_argument("--culture", help="Alias of the culture", required=True)
    argParser.add_argument("--id", help="Id of the culture", required=True)

    args = argParser.parse_args()

    MachineLearningPipeline().start_pipeline(
        args.culture,
        args.id
    )
