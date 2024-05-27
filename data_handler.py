import pandas as pd

from datetime import datetime
import yfinance as yf


class DataHandler:

    def __init__(self):
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        self.snp500 = tables[0]

        self.start_date = datetime(2015, 1, 1)
        self.end_date = datetime.today().date()

    def get_snp(self):

        return self.snp500

    def get_data(self, ticker):
        data = yf.download(
            ticker, start=self.start_date, end=self.end_date, progress=False
        )
        data = data.reset_index()
        return data

    def __len__(self):
        return len(self.snp500)

    def get_unique(self, col):

        return self.snp500[col].value_counts().reset_index()
