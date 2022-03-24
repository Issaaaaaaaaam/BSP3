from matplotlib import pyplot as plt 
from matplotlib import style 
import pandas as pd 
import numpy as np 

#function to fill all the NaN in the resulting tables we have. 
def fillna_downbet(df):
    df = df.copy()
    for col in df:
        non_nans = df[col][~df[col].apply(np.isnan)]
        start, end = non_nans.index[0], non_nans.index[-1]
        df[col].loc[start:end] = df[col].loc[start:end].fillna(method='ffill')
    return df

df = pd.read_csv("1.csv")
ds = pd.read_csv("2.csv")
#sort the dates and define them as indexes in the table

def prepare (df): 
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index("date")
    df = df.sort_index()

prepare(df)
prepare(ds)

#Row by row the program isolates the CAR's for each bank and append them to a new table where the CAR of a same bank are grouped in a column. 



#Plotting part of the code. 

ax = df.plot.scatter(x = 'S&P500_ret' , y = 'CAR' , figsize = (10,10), color = 'blue', label = "Banks with a CAR superior to 8 (adjusted)")
ds.plot.scatter(x = 'S&P500_ret' , y = 'CAR' , figsize = (10,10), color = 'red',label = "Banks with a CAR inferior to 8 (adjusted)", ax = ax )

ax.set_title("The correlation of the measured CAR of a bank by S&P500 quarterly returns.")

# Shrink current axis by 20%
'''box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))'''
plt.show()
