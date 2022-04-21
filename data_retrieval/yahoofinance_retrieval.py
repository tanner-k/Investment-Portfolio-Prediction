import requests
import pandas as pd
import numpy as np
import time

# yf_key = "A9n29cZYNj7xPDyy1TCcX1XHl7ogCWjM3yGH6YYn" #tk2asian@gmail.com

yf_key = "ZYC7tMe2QD7ib5bLtLCDn1P6ws8QpPua9389FeqZ"  #tanner.kunz22@gmail.com

headers = {
    "x-api-key": yf_key
}

sp100 = pd.read_csv("data/S&P100.csv", index_col=0)

data = pd.DataFrame()

bins = np.linspace(0, len(sp100), 11, dtype=int)

for i in range(len(bins)):

    try:
        b = sp100.iloc[bins[i]:bins[i + 1]]
        s = []
        for j in b.values.tolist():
            for k in j:
                s.append(k)

        symbols = ",".join(s)

        querystring = {
            "symbols": symbols,
            "interval": "1d",
            "range": "10y"
        }

        bin_df = pd.DataFrame()

        url = f"https://yfapi.net/v8/finance/spark"
        response = requests.request("GET", url, headers=headers, params=querystring)
        json_dict = response.json()

        for ticker in json_dict.keys():

            try:
                close = json_dict[ticker]["close"]
                date = json_dict[ticker]["timestamp"]
                col = [json_dict[ticker]["symbol"]]

                df = pd.DataFrame(close, index=date, columns=col)
                bin_df = pd.concat([bin_df, df], axis=1)
            except TypeError:
                continue

        data = pd.concat([data, bin_df], axis=1)

    except IndexError:
        continue

data.to_csv("data/StockData_10year.csv")