import pandas as pd
import csv
df=pd.read_csv(r'C:\Users\\28708\\Desktop\\data_file\\diabetic_data.csv', header=None)
data_e=df.sample(n=30000)
print (data_e)
data_e.to_csv(r'C:\Users\\28708\\Desktop\\data_file\\diabetic_data_int1.csv', index=None,header=0)