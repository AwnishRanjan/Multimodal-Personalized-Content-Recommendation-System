import requests 
import pandas as pd 

news_api = 'af35ad3f066a4d27b02c9fa758cb6483'
news_url = 'https://newsapi.org/v2/everything'

params = {
    'q':'technology',
    'apikey':news_api,
    'pageSize': 100
}

response = requests.get(url=news_url , params=params)
articles = response.json()['articles']

df = pd.DataFrame(articles)
df.to_csv('data/raw/text_data.csv',index=False)



