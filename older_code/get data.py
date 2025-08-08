import requests
import pandas as pd

API_KEY = "579b464db66ec23bdd000001b608636e55b1459450beeffffae9caae"
# API_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"
BASE = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"

limit = 100000
offset = 0
data = []
i = 0
while (True):
    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": limit,           
        "filters[Commodity]" : "Groundnut" , 
        "offset": offset,      
        "filters[District]" : "Rajkot" , 
        "filters[State]" : "Gujarat"
   
    }

    response = requests.get(BASE, params=params)
    if not response:
        break
    print(response)
    response_json = response.json()
    records = response_json["records"]
    
    data.extend(records)
    print(len(data))
    offset += limit
    i += 1
    print(i)
    if i> 100000:
        break
# print(data)
data_df = pd.DataFrame(data)
print(data)
data_df.to_csv('sample_data_Rajkot3.csv')
# print(data)