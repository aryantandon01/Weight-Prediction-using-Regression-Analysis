import pandas as pd
import numpy as np
from sklearn import linear_model
df = pd.read_csv (r"C:\Users\aryan\Desktop\WeightFinal.csv")
df
reg = linear_model.LinearRegression()
reg.fit(df[['ht_cm','arm_mm','hl_mm','fl_mm','lel_mm','nc_mm','bust_mm','wc_mm','hw_mm']],df.wt_kg)
reg.coef_
reg.intercept_
reg.predict([[179.01,830.62,183.57,246.21,1012.95,350.41,858.51,760.24,557.80]])
import seaborn as sns
sns.set_style('white')
sns.set_style('ticks')
sns.regplot(x='wt_kg', y='Predicted weight', data=df)