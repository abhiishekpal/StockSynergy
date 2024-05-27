import pandas as pd

from datetime import datetime
from utils import cutoff_point


class FeatureEngineering:

    def __init__(self):
        self.diff_year = 2
        self.tracking_column = "Adj Close"
        self.windows = [7, 60]
        self.start_date = datetime(2015, 1, 1)
        self.end_date = datetime.today().date()

    def fill_missing_date(self, data):
        data["Date"] = pd.to_datetime(data["Date"])
        all_dates_in_range = pd.date_range(start=self.start_date, end=self.end_date)
        data = data.set_index("Date").reindex(all_dates_in_range)
        for col in data.columns:
            data[col] = data[col].fillna(method="ffill")
        data = data.reset_index()
        return data

    def get_price_diff(self, data):

        data["shifted_Date"] = data["index"] + pd.DateOffset(years=self.diff_year)
        data1 = data[["index", self.tracking_column, "shifted_Date"]]
        data1.columns = ["date", "price", "future_date"]
        data2 = data[["index", self.tracking_column]]
        data2.columns = ["date", "future_price"]
        joined_data = pd.merge(
            data1, data2, left_on="future_date", right_on="date", how="inner"
        )[["date_x", "price", "future_date", "future_price"]]
        joined_data.columns = ["current_date", "price", "future_date", "future_price"]

        joined_data["price_diff"] = (
            joined_data["future_price"] - joined_data["price"]
        ) / joined_data["price"]

        return joined_data

    def _get_movement(self, x):
        if abs(x) < 0.05:
            return "Flat"
        if x < 0:
            return "Down"
        else:
            return "Up"

    def _get_uncertainity(self, x, window):

        if (
            x[f"cutoff_point_{window}_low"]
            < x["price_diff"]
            < x[f"cutoff_point_{window}_high"]
        ):
            return 1
        return 0

    def _get_movement_type(self, x):

        if x["in_7"] == 1 and x["in_60"] == 1:
            return 0

        if x["in_7"] == 1 and x["in_60"] == 0:
            return 1

        if x["in_7"] == 0 and x["in_60"] == 1:
            return 2

        if x["in_7"] == 0 and x["in_60"] == 0:
            return 3

    def add_rolling_average(self, data):

        for window in self.windows:
            data[f"mean_diff_{window}"] = (
                data["price_diff"].rolling(window=window).mean()
            )
            data[f"std_diff_{window}"] = (
                data["price_diff"].rolling(window=window).mean()
            )
        for window in self.windows:
            data[f"cutoff_point_{window}_high"] = data.apply(
                lambda x: x[f"mean_diff_{window}"]
                + cutoff_point(x[f"mean_diff_{window}"], x[f"std_diff_{window}"], 80),
                axis=1,
            )
            data[f"cutoff_point_{window}_low"] = data.apply(
                lambda x: x[f"mean_diff_{window}"]
                - cutoff_point(x[f"mean_diff_{window}"], x[f"std_diff_{window}"], 80),
                axis=1,
            )

        return data

    def get_feature(self, data):

        temp = data.dropna()
        data["movement"] = data["price_diff"].apply(lambda x: self._get_movement(x))
        for window in self.windows:
            data[f"in_{window}"] = data.apply(
                lambda x: self._get_uncertainity(x, window), axis=1
            )
        data["movement_type"] = data.apply(lambda x: self._get_movement_type(x), axis=1)

        return data
