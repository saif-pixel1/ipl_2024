# %%
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt 
month_map = {
    'January': 1, 'February': 2, 'March': 3,
    'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9,
    'October': 10, 'November': 11, 'December': 12
}
import datetime

# %%
df2 = pd.read_csv("IPL.csv")

# %%
df2.head()

# %%
df2["Date"]= df2["date"].str.split(" ").str[1].str.split(",").str[0]
df2["year"]= df2["date"].str.split(" ").str[1].str.split(",").str[1]
df2["month"]= df2["date"].str.split(" ").str[0]

# %%
df2["Date"].fillna(df2["Date"].mode()[0],inplace=True)
df2["year"].fillna(df2["year"].mode()[0],inplace=True)
df2["Date"]=df2["Date"].astype(int)
df2["year"]=df2["year"].astype(int)

# %%
df3 = pd.get_dummies(df2["month"],dtype=int)

# %%
df2["venue"] = df2["venue"].str.split(",").str[1]


# %%
df2["best_bowling_wickets"] = df2["best_bowling_figure"].str.split("--").str[0].astype(int)
df2["best_bowling_runs_given"] = df2["best_bowling_figure"].str.split("--").str[1].astype(int)

# %%


# %%
df2['month_num'] = df2['month'].map(month_map)

# %%
df = pd.DataFrame(df2["match_winner"])

# %%
df2.drop(["date", "best_bowling_figure", "match_id","month","match_winner" ], axis=1, inplace=True)

# %%
df2 = pd.get_dummies(df2)
df2 = df2.fillna(0).astype(int)

# %%
df2.astype(int)

# %%
df

# %%
x = df2


# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(df['match_winner'])  # single column

# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# %%
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)


# %%
y_pred

# %%
y_pred_labels = le.inverse_transform(y_pred)

# %%
y_pred_labels


# %%
y_test_labels = le.inverse_transform(y_test)

# %%
score = pd.DataFrame({"Actual": y_test_labels, "Predicted": y_pred_labels})

# %%
score

# %%
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# %%
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
