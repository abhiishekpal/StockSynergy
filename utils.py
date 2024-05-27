from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import scipy.stats as stats
import plotly.graph_objects as go


path = "./"
## Note: uncomment this if working with colab
# from google.colab import drive
# drive.mount('/content/drive')
# path = "/content/drive/My Drive"


label_index = {'Up0': 1, 'Down3': 2, 'Up1': 3, 'Flat3': 4, 'Up3': 5, 'Flat0': 6, 'Flat2': 7, 'Flat1': 8, 'Up2': 9, 'Down2': 10, 'Down0': 11, 'Down1': 12}

def cutoff_point(mean, std_dev, percent_away):
    percent_decimal = percent_away / 200
    z_score = stats.norm.ppf(0.5 + percent_decimal)
    point = z_score * std_dev

    return point

def plot_lines(data, colx, coly):
    fig = go.Figure()

    for col in coly:
        fig.add_trace(go.Scatter(x=data[colx], y=data[col], mode='lines', name=col))
    fig.update_layout(
        xaxis_title=colx,
        yaxis_title='% diff',
        title='Rate Change Plot'
    )
    fig.show()


def get_data_label(start_date=datetime(2015, 1, 1), end_date=datetime(2021, 1, 1)):
  df = pd.read_csv('data_snp500_movement_v2.csv')
  df["current_date"] = pd.to_datetime(df["current_date"])
  df = df.loc[(df["current_date"]>start_date) & (df["current_date"]<end_date)]
  # -1 here signifies that the stock label is not present for this point
  df["label"] = df["movement"].astype(str) + df["movement_type"].astype(str)
  new_label = dict(df["label"].value_counts())
  df["label_index"] = df["label"].apply(lambda x: label_index.get(x))
  temp = df.copy()
  temp['has_missing_values'] = temp.isnull().any(axis=1)
  temp.loc[temp["has_missing_values"]==True, "has_missing_values"] = -1
  temp.loc[temp["has_missing_values"]==False, "has_missing_values"] = 1
  temp["label_index"] *= temp["has_missing_values"]
  temp = temp[["Ticker", "current_date", "label_index"]]

  return temp, df

def get_length_of_tickers(temp):
  length = [len(temp.loc[temp["Ticker"]==ticker]) for ticker in temp["Ticker"].unique()]
  print("Unique Lengths: ", set(length)) # checking for any issue in feature preparation, all dataframe should be of same length
  length = length[0]
  return length

def get_array(temp):
  return np.array(temp["label_index"])

def get_similarity_score(dft, arr, ls):

  all_tickers = dft["Ticker"].unique().tolist()
  scores = np.zeros((len(all_tickers), len(all_tickers)))
  for i, ticker1 in tqdm(enumerate(all_tickers)):
    base = arr[i*ls:(i+1)*ls]
    for j, ticker2 in enumerate(all_tickers):
      target = arr[j*ls:(j+1)*ls]
      same_mask = (base == target).astype(int)
      mask_base = (base>0).astype(int)
      mask_target = (target>0).astype(int)
      remove_mask = mask_base * mask_target
      same_mask = same_mask * remove_mask
      matches = np.sum(same_mask)
      total = np.sum(remove_mask)
      similarity = matches/(total+1)
      scores[i, j] = similarity
  return scores, all_tickers


def save_data(data, name):
    if ".csv" in name:
       data.to_csv(f'{path}/{name}', index=False)
    elif ".npy" in name:
       np.save(f'{path}/{name}', data)
    elif ".pickle" in name:
        with open(f'{path}/{name}', 'wb') as f:
            pickle.dump(data, f)


def load_data(name):
    if ".csv" in name:
       data =pd.read_csv(f'{path}/{name}')
    elif ".npy" in name:
       data = np.load(f'{path}/{name}')
    elif ".pickle" in name:
        with open(f'{path}/{name}', 'rb') as f:
            data = pickle.load(f)
    return data

def sum_except_diagonal(arr):
    arr = np.array(arr)
    diagonal_mask = np.eye(arr.shape[0], dtype=bool)
    sum_except_diag = arr[~diagonal_mask].sum()
    return sum_except_diag
    
