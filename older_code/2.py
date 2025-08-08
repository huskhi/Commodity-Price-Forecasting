import pandas as pd
# data = pd.read_csv('sample_data_Junagarh.csv' )
data  =  pd.read_csv('sample_data_Rajkot2.csv' )

# print(data.District.unique())
# print(data[data['District'] == 'Rajkot'].Arrival_Date.min())
# print(data[data['District'] == 'Rajkot'].Arrival_Date.max())


# data = data[data['District'] == 'Junagarh']

print(len(data))
# print(data['Market'].value_counts().head(5))

data = data[data['Market'] == 'Rajkot' ]

print(data)