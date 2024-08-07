import os
import logging
import requests
import xlrd
import json
import pandas as pd
from bs4 import BeautifulSoup
from unidecode import unidecode


class DatasetPipeline:
    # Requests Configurations
    base_url = "https://www.cepea.esalq.usp.br/br"
    cookies = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/109.0.0.0 Safari/537.36 "
    }

    # Directories Configurations
    root_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = f"{root_dir}/datasets"

    # Options Configurations
    exclude_cultures = ["algodao", "mandioca", "ovos", "suino", "tilapia"]

    # Datasets Configurations
    time_steps = None

    def getter_datasets_json(self, update=False) -> None:
        if update or not os.path.exists(f"{self.root_dir}/cultures.json"):
            cultures = self.getter_dataset_list()

            with open(f"{self.root_dir}/cultures.json", "w", encoding="utf-8") as f:
                json.dump(cultures, f, indent=4)
        else:
            with open(f"{self.root_dir}/cultures.json", "r", encoding="utf-8") as f:
                cultures = json.loads(f.read())

        return cultures

    def getter_dataset_list(self):
        logging.info("Retrieving dataset list")

        self.create_datasets_directories()
        cultures = {}

        start_request = requests.get(self.base_url, headers=self.cookies)
        start_response = BeautifulSoup(start_request.content, "html.parser")

        categories = [
            culture["href"]
            for culture in start_response.select("div#imagenet-categoria div ul li a")
            if "/indicador/" in culture["href"]
        ]

        for i, category in enumerate(categories):
            logging.info(
                "Retrieving culture details {}/{}".format(i + 1, len(categories))
            )

            culture, sub_cultures = self.update_details_cultures(category)
            if culture not in self.exclude_cultures and sub_cultures:
                cultures[culture] = sub_cultures

        return cultures

    def update_details_cultures(self, base_url):
        sub_cultures = {}

        request = requests.get(base_url, headers=self.cookies)
        response = BeautifulSoup(request.content, "html.parser")

        for culture in response.select(
            "div.imagenet-content.imagenet-left div.imagenet-col-12"
        ):
            block_url = culture.select_one("a:nth-of-type(4)")
            if block_url:
                url = block_url["href"]
                title = culture.select_one(
                    "div.imagenet-col-8.imagenet-sm-12.imagenet-table-titulo"
                ).text
                block_id = url.split("?id=")[-1]

                if block_id not in sub_cultures:
                    sub_cultures.update(
                        {
                            block_id: {
                                "url": url,
                                "title": unidecode(title),
                            }
                        }
                    )

        culture_alias = base_url.split("/indicador/")[1].split(".")[0]

        return culture_alias, sub_cultures

    def create_datasets_directories(self):
        logging.info("Creating dataset directories")

        if not os.path.exists(self.datasets_dir):
            os.makedirs(self.datasets_dir)

    def getter_dataset(self, culture_alias, culture_id, time_steps):
        logging.info("Dataset Getting")
        self.time_steps = time_steps
        self.create_datasets_directories()

        filename = f"{culture_alias}_{culture_id}_dataset.xls"

        if filename not in os.listdir(f"{self.datasets_dir}"):
            resp = requests.get(
                f"{self.base_url}/indicador/series/{culture_alias}.aspx?id={culture_id}",
                headers=self.cookies,
            )

            output = open(f"{self.datasets_dir}/{filename}", "wb")
            output.write(resp.content)
            output.close()

        workbook = xlrd.open_workbook(
            f"{self.datasets_dir}/{filename}", ignore_workbook_corruption=True
        )

        return pd.read_excel(
            workbook,
            sheet_name="Plan 1",
            skiprows=range(0, 3),
        )

    def set_regressor_attributes(self, df, attribute) -> pd.DataFrame:
        logging.info("Add time steps to the dataset")

        list_of_prev_t_instants = list(range(1, self.time_steps))

        list_of_prev_t_instants.sort()
        start = list_of_prev_t_instants[-1]
        end = len(df)
        df.reset_index(drop=True)

        df_copy = df[start:end]
        df_copy.reset_index(inplace=True, drop=True)

        df_temp = pd.DataFrame()

        for prev_t in list_of_prev_t_instants:
            new_col = pd.DataFrame(
                df[attribute].iloc[(start - prev_t) : (end - prev_t)]
            )
            new_col.reset_index(drop=True, inplace=True)
            new_col.rename(
                columns={attribute: "{}_(t-{})".format(attribute, prev_t)}, inplace=True
            )

            df_temp = pd.concat([df_temp, new_col], sort=False, axis=1)

        df_copy = pd.concat([df_copy, df_temp], sort=False, axis=1)

        return df_copy

    def dataset_config(self, df):
        logging.info("Renaming and deleting columns from the dataset")
        df.rename(columns={"Data": "date", "À vista R$": "value"}, inplace=True)
        df.drop(columns=["À vista US$"], inplace=True)

        logging.info("Formatting dataset columns")
        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
        df["value"] = df["value"].apply(pd.to_numeric)

        logging.info("Add new columns to the dataset")
        df["seasons"] = df["date"].apply(lambda x: self.set_year_seasons(x))

        df = self.set_regressor_attributes(df, "value")

        return df

    @staticmethod
    def dataset_predict_skip(df, new_values):
        df_columns = df.columns

        df = df.tail(1)

        if new_values:
            new_df = df.values.tolist()
            new_df[0].insert(1, new_values[-1])
            new_df[0].pop(-1)

            df = pd.DataFrame(new_df, columns=df_columns)

        return df

    @staticmethod
    def dataset_splitting(df):
        logging.info("Dataset getting X, y")

        df = df.drop(["date"], axis=1)

        return df.drop(["value"], axis=1), df.loc[:, "value"]

    @staticmethod
    def set_year_seasons(row):
        # 0 - Autumn | 1 - Winter | 2 - Spring | 3 - Summer

        day = int(row.day)
        month = int(row.month)

        dict_ = {
            1: 3,
            2: 3,
            3: [3, 0],
            4: 0,
            5: 0,
            6: [0, 1],
            7: 1,
            8: 1,
            9: [1, 2],
            10: 2,
            11: 2,
            12: [2, 3],
        }
        season = dict_[month]

        if type(season) == int:
            return season

        if day < 21:
            return season[0]
        else:
            return season[1]
