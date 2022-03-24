from matplotlib import pyplot as plt 
from matplotlib import style 
import pandas as pd 
import numpy as np 



df = pd.read_csv("CAR_script_results.csv")

bank = list(df['Bank_id'].unique())

inp = int(input("Enter the ID of the bank you are searching: "))

df['date'] = pd.to_datetime(df['date'])

if inp not in bank: 
    raise Exception("You did not enter a valid bank ID")

for id in df["Bank_id"].unique(): 
    if id == inp: 
        id_df = df.copy()[df["Bank_id"] == id]
        id_df = id_df.rename(columns={'CAR': f'{id}_CAR'})
        id_df = id_df.set_index("date")
        id_df = id_df.sort_index()
        print(id_df)
        
id_df = id_df.drop('Bank_id', 1)
fig = id_df.plot()
# Shrink current axis by 20%
box = fig.get_position()
fig.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
