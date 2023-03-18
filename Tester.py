# import necessary libraries
import pandas as pd
import seaborn as sns

sns.set_theme()

# import the dataset
df = pd.read_csv('car-price.csv')

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
cate_cols = list(df.select_dtypes(["object", "bool"]).columns)
for col in cate_cols:
    df[col] = encoder.fit_transform(df[col])

def melt(data_frame, col, data: dict, inplace: bool = False):
    lst = []
    if inplace:
        df = data_frame
    else:
        df = {}
    for da in data_frame[col]:
        if da in data:
            lst.append(data[da])
    for info in data:
        vals = []
        for i in range(len(lst)):
            vals.append(1 if lst[i] == data[info] else 0)
        df[data[info]] = vals
    return df

data = {0: 'Automatic', 1: 'Manual', 2: 'Triptronic', 3: 'Variator'}
print(melt(df, 'Gear box type', data))
