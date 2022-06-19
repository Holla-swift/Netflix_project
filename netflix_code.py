## Analysis of Netflix Movies and Series
import os
import pandas as pd
import numpy as np
from dateutil import parser

import matplotlib.pyplot as plt
import seaborn as sns

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from wordcloud import WordCloud


### File Path and Data Extraction
directory = os.getcwd()
file_path = directory + '/netflix_titles.csv'
netflix_data = pd.read_csv(file_path)


netflix_data = netflix_data.fillna(0)
data = netflix_data.copy()
data['dateAdded'] = pd.to_datetime(data['date_added'])
data['monthAdded'] = pd.DatetimeIndex(data['dateAdded']).month
data['yearAdded'] = pd.DatetimeIndex(data['dateAdded']).year
data['monthName'] = data['dateAdded'].apply(lambda x: x.strftime("%B")) 
data.head(2)


movies_rating = data.groupby('rating').count()
movies_rating = pd.DataFrame(movies_rating.show_id)
movies_rating = movies_rating.iloc[1:]
plt.bar(movies_rating.index, movies_rating.show_id, color='red')
plt.title('Rating analysis of  Netflix Videos ', size=14)
plt.xlabel('Rating category')
plt.ylabel('No of Movies')
plt.xticks(rotation=90)
plt.show()


# yearly release
yearData = data.groupby('yearAdded').count()
yearData = pd.DataFrame(yearData.show_id)
plt.plot(yearData.index, yearData.show_id, color='red')
plt.title('No of Netflix Videos per Annum', size=14)
plt.xlabel('No of movies')
plt.ylabel('Year')
plt.show()


monthData = data.groupby('monthName').count()
monthData = pd.DataFrame(monthData.show_id)
months = [ 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
monthData = monthData.reindex(months)
plt.bar(monthData.index, monthData.show_id, color='red')
plt.title('No of Videos Added to Netflix Per Month', size=16)
plt.ylabel('No of movies')
plt.xlabel('Month')
plt.xticks(rotation = 90)
plt.show()


### Netflix videos breakdown
set(netflix_data.type)
netflix_movies = data[data.type == 'Movie']
netflix_series = data[data.type == 'TV Show']
netflix_type = data.groupby('type').count()
netflix_type = pd.DataFrame(netflix_type.show_id)
plt.barh(netflix_type.index, netflix_type.show_id)
plt.title('No of Videos Per Category', size=14)
plt.ylabel('No of movies')
plt.xlabel('Category')
plt.show()


release_year = data.groupby('release_year').count()
release_year = pd.DataFrame(release_year.show_id)
plt.plot(release_year.index, release_year.show_id)
plt.title('No of Netflix videos released', size=14)
plt.xlabel('No of movies')
plt.ylabel('Year')
plt.show()


#### Netflix Movies Analysis
netflix_movies.sample(2)
# yearly release
yearData1 = netflix_movies.groupby('yearAdded').count()
yearData1 = pd.DataFrame(yearData1.show_id)
plt.plot(yearData1.index, yearData1.show_id, color='blue')
plt.title('No of Netflix Movies Added per year', size=14)
plt.xlabel('No of movies')
plt.ylabel('Year')
plt.show()


monthData1 = netflix_movies.groupby('monthName').count()
monthData1 = pd.DataFrame(monthData1.show_id)
months = [ 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
monthData1 = monthData1.reindex(months)
plt.bar(monthData1.index, monthData1.show_id, color='blue')
plt.title('No of Videos Added to Netflix Per Month', size=16)
plt.ylabel('No of movies')
plt.xlabel('Month')
plt.xticks(rotation = 90)
plt.show()


new  = netflix_movies["duration"].str.split(" ", n=1, expand=True)
netflix_movies["Dmins"] = new[0]
netflix_movies['Dmins'] = pd.to_numeric(netflix_movies['Dmins'])
netflix_movies = netflix_movies.fillna(0)
netflix_movies
netflix_movies = netflix_movies.drop(['minsDuration'], axis=1)
sns.histplot(data=netflix_movies, x='Dmins')


### Netflix Series Analysis
netflix_series.head(2)
# yearly release
yearData2 = netflix_series.groupby('yearAdded').count()
yearData2 = pd.DataFrame(yearData2.show_id)
plt.plot(yearData2.index, yearData2.show_id, color='g')
plt.title('No of Netflix Series Added per year', size=14)
plt.xlabel('No of Series')
plt.ylabel('Year')
plt.show()


monthData2 = netflix_series.groupby('monthName').count()
monthData2 = pd.DataFrame(monthData2.show_id)
months = [ 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
monthData2 = monthData2.reindex(months)
plt.bar(monthData2.index, monthData2.show_id, color='g')
plt.title('No of Series Added to Netflix Per Month', size=16)
plt.ylabel('No of movies')
plt.xlabel('Month')
plt.xticks(rotation = 90)
plt.show()


seasonal_movies = netflix_series.groupby('duration').count()
seasonal_movies = pd.DataFrame(seasonal_movies.show_id)
seasons = [ '1 Season', '2 Seasons', '3 Seasons', '4 Seasons', '5 Seasons', '6 Seasons', '7 Seasons', '8 Seasons', '9 Seasons', '10 Seasons', '11 Seasons', '12 Seasons', '13 Seasons', '15 Seasons', '17 Seasons']
seasonal_movies = seasonal_movies.reindex(seasons)
plt.bar(seasonal_movies.index, seasonal_movies.show_id, color='r')
plt.title('Seasonal distribution of Netflix series', size=16)
plt.ylabel('Number of series')
plt.xlabel('Season(s)')
plt.xticks(rotation = 90)
plt.show()


stop_words = set(stopwords.words('english'))
netflix_series['title_no_stopwords'] = netflix_series['title'].apply(lambda x: [item for item in str(x).split() if item not in stop_words])

all_words = list([a for b in netflix_series['title_no_stopwords'].tolist() for a in b])
all_words_str = ' '.join(all_words) 

def plot_cloud(wordcloud):
    plt.figure(figsize=(30, 20))
    plt.imshow(wordcloud) 
    plt.axis("off");

wordcloud = WordCloud(width = 500, height = 250, random_state=1, background_color='white', 
                      colormap='viridis', collocations=False).generate(all_words_str)
plot_cloud(wordcloud)


stop_words = set(stopwords.words('english'))
data['title_no_stopwords'] = data['description'].apply(lambda x: [item for item in str(x).split() if item not in stop_words])

all_words = list([a for b in data['title_no_stopwords'].tolist() for a in b])
all_words_str = ' '.join(all_words) 

def plot_cloud(wordcloud):
    plt.figure(figsize=(30, 20))
    plt.imshow(wordcloud) 
    plt.axis("off");

wordcloud = WordCloud(width = 500, height = 250, random_state=1, background_color='white', 
                      colormap='viridis', collocations=False).generate(all_words_str)
plot_cloud(wordcloud)
stop_words = set(stopwords.words('english'))
data['title_no_stopwords'] = data['country'].apply(lambda x: [item for item in str(x).split() if item not in stop_words])

all_words = list([a for b in data['title_no_stopwords'].tolist() for a in b])
all_words_str = ' '.join(all_words) 

def plot_cloud(wordcloud):
    plt.figure(figsize=(30, 20))
    plt.imshow(wordcloud) 
    plt.axis("off");

wordcloud = WordCloud(width = 500, height = 250, random_state=1, background_color='white', 
                      colormap='viridis', collocations=False).generate(all_words_str)
plot_cloud(wordcloud)