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

# Converting Label Name to numerical values
label_index = {
    "Up0": 1,
    "Down3": 2,
    "Up1": 3,
    "Flat3": 4,
    "Up3": 5,
    "Flat0": 6,
    "Flat2": 7,
    "Flat1": 8,
    "Up2": 9,
    "Down2": 10,
    "Down0": 11,
    "Down1": 12,
}


def cutoff_point(mean, std_dev, percent_away):
    """
    For any std_dev computes the point where a given percentile will lie from the mean
    """
    percent_decimal = percent_away / 200
    z_score = stats.norm.ppf(0.5 + percent_decimal)
    point = z_score * std_dev

    return point


def plot_lines(data, colx, coly):
    """
    Plot a particular line plot
    """
    fig = go.Figure()

    for col in coly:
        fig.add_trace(go.Scatter(x=data[colx], y=data[col], mode="lines", name=col))
    fig.update_layout(xaxis_title=colx, yaxis_title="% diff", title="Rate Change Plot")
    fig.show()


def get_data_label(start_date=datetime(2015, 1, 1), end_date=datetime(2021, 1, 1)):
    """
    Filters the complete prepared data in the provided date range
    and creates the final numerical label for each event in a stock
    Input:
        start_date: start_date for consideration
        end_date: end_date to cutoff at
    Returns:
        temp: new dataframe with ticker, date and its label
        df: the complete dataframe information in a given date range
    """

    df = pd.read_csv("data_snp500_movement_v2.csv")
    df["current_date"] = pd.to_datetime(df["current_date"])
    df = df.loc[(df["current_date"] > start_date) & (df["current_date"] < end_date)]
    
    # combining columns of label to derive one column and converting that to a numerical representation
    df["label"] = df["movement"].astype(str) + df["movement_type"].astype(str)
    df["label_index"] = df["label"].apply(lambda x: label_index.get(x))

    temp = df.copy()
    temp["has_missing_values"] = temp.isnull().any(axis=1)
    # -1 here is used to signify that the stock label is not present for this point
    temp.loc[temp["has_missing_values"] == True, "has_missing_values"] = -1
    temp.loc[temp["has_missing_values"] == False, "has_missing_values"] = 1
    temp["label_index"] *= temp["has_missing_values"]
    temp = temp[["Ticker", "current_date", "label_index"]]

    return temp, df


def get_length_of_tickers(temp):
    """
    Check length of each stock sequence and verifies if they are same
    Input:
        temp: dataframe
    Returns:
        length: sequence length
    """

    length = [
        len(temp.loc[temp["Ticker"] == ticker]) for ticker in temp["Ticker"].unique()
    ]
    print(
        "Unique Lengths: ", set(length)
    )  # checking for any issue in feature preparation, all dataframe should be of same length
    length = length[0]
    return length


def get_array(temp):
    """
    Returns the vector representation of required column
    """
    return np.array(temp["label_index"])


def get_similarity_score(dft, arr, ls):

    """
    Computes the similarity between each stock sequence
    Input:
        arr: array representation of the stock sequence (for easier slicing)
        dft: dataframe consisting the ticker
        ls: list with sequence length for each stock
    Output:
        scores: similarity matrix
        all_tickers: ticker list
    """

    all_tickers = dft["Ticker"].unique().tolist()
    scores = np.zeros((len(all_tickers), len(all_tickers)))
    for i, ticker1 in tqdm(enumerate(all_tickers)):
        # vector we are going to compare rest against
        base = arr[i * ls : (i + 1) * ls]
        for j, ticker2 in enumerate(all_tickers):
            # current vector under consideration
            target = arr[j * ls : (j + 1) * ls]
            # find indexes where base and target are same
            same_mask = (base == target).astype(int)
            # mask to not consider indexes where mask value was less than 0 since we have filled nan values with -1
            mask_base = (base > 0).astype(int)
            mask_target = (target > 0).astype(int)
            # If any of the two mask index is False we will consider those to be removed from considering
            remove_mask = mask_base * mask_target
            # apply the remove mask
            same_mask = same_mask * remove_mask
            # computing the total % of times when the two sequence overlapps
            matches = np.sum(same_mask)
            total = np.sum(remove_mask)
            similarity = matches / (total + 1)
            scores[i, j] = similarity
    return scores, all_tickers


def save_data(data, name):
    """
    Save the respective files
    """

    if ".csv" in name:
        data.to_csv(f"{path}/{name}", index=False)
    elif ".npy" in name:
        np.save(f"{path}/{name}", data)
    elif ".pickle" in name:
        with open(f"{path}/{name}", "wb") as f:
            pickle.dump(data, f)


def load_data(name):
    """
    Load the respective file
    """
    if ".csv" in name:
        data = pd.read_csv(f"{path}/{name}")
    elif ".npy" in name:
        data = np.load(f"{path}/{name}")
    elif ".pickle" in name:
        with open(f"{path}/{name}", "rb") as f:
            data = pickle.load(f)
    return data


def sum_except_diagonal(arr):
    """
    Sum of elements in a 2d array barring the diagonal
    """
    arr = np.array(arr)
    diagonal_mask = np.eye(arr.shape[0], dtype=bool)
    sum_except_diag = arr[~diagonal_mask].sum()
    return sum_except_diag
