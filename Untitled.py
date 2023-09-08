#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries


# In[2]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as seab
import plotly.express as px
from wordcloud import WordCloud
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score as score
from sklearn.tree import DecisionTreeRegressor


# In[3]:


# Read Data


# In[4]:


df = pd.read_csv('D:/data science/movie/Movies.csv', encoding='latin-1')


# In[5]:


df


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.dtypes


# In[9]:


df.isnull().sum()


# In[10]:


df.isnull().values.any()


# In[11]:


df.isnull().sum()


# In[12]:


seab.heatmap(df.isnull())


# In[13]:


df.dropna(axis=0, inplace =True)


# In[14]:


df.duplicated().any()


# In[15]:


df.drop_duplicates()


# In[16]:


df.describe() #numerical


# In[17]:


df.describe(include = 'all') #all data


# In[18]:


df['Duration'] = df['Duration'].str.extract('(\d+)')
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
#df['Duration']= df['Duration'].str.replace('min','').apply(pd.to_numeric)


# In[19]:


df["Votes"]=df["Votes"].replace("$5.16M", 516)
df["Votes"] = pd.to_numeric(df['Votes'].str.replace(',',''))


# In[20]:


#removing the paranthesis 
df['Year'] = df['Year'].str.extract('(\d+)')  # Extract numeric part of the string
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')  # Convert to numeric
#df['Year'] = df['Year'].str.replace('(','').str.replace(')','')


# In[21]:


genre=df['Genre']
genres=df['Genre'].str.split(',',expand=True)

from collections import Counter

# Assuming 'genres' is a DataFrame column or Series containing genre values
genre_counts = Counter(genre for genre in genres.values.flatten() if genre is not None)

# Sort the genre counts alphabetically
sorted_genre_counts = dict(sorted(genre_counts.items()))

for genre, count in sorted_genre_counts.items():
    print(f"{genre}: {count}")


# In[22]:


genresPie = df['Genre'].value_counts()
genresPie.head(5)


# In[23]:


directors = df["Director"].value_counts()
directors.head(5)


# In[24]:


actors = pd.concat([df['Actor 1'], df['Actor 2'], df['Actor 3']]).dropna().value_counts()
actors.head(5)


# In[25]:


#Visulization


# In[26]:


#replacing null values with the most common rating given to an Indian Movie
df['Rating'].fillna(df['Rating'].mode().max(),inplace=True)

#replacing null values with average duration of a movie in India
df['Duration'].fillna(df['Duration'].mean(),inplace=True)

#replacing null values with average votes recived by a Movie
df['Votes'].fillna(df['Votes'].mean(),inplace=True)

# Drop rows with NaN values in the 'Year' column
df.dropna(subset=['Year'], inplace=True)


# In[27]:


# Convert the 'Year' column to integers
df['Year'] = df['Year'].astype(int)


# In[28]:


ax = seab.lineplot(data=df['Year'].value_counts().sort_index())
tick_positions = range(int(min(df['Year'])), int(max(df['Year'])) + 1, 5)
ax.set_title("Annual Movie Release Counts Over Time")
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_positions, rotation = 90)
ax.set_xlabel("Years")
ax.set_ylabel("Count")
plt.show()


# In[29]:


ax = seab.boxplot(data=df, y='Year')
ax.set_ylabel('Year')
ax.set_title('Box Plot of Year')
plt.show()


# In[30]:


ax = seab.lineplot(data=df.groupby('Year')['Duration'].mean().reset_index(), x='Year', y='Duration')
tick_positions = range(min(df['Year']), max(df['Year']) + 1, 5)
ax.set_title("Average Movie Duration Trends Over the Years")
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_positions, rotation = 90)
ax.set_xlabel("Years")
ax.set_ylabel('Average Duration(in minutes)')
plt.show()


# In[31]:


ax = seab.boxplot(data=df, y='Duration')
ax.set_title("Box Plot of Average Movie Durations")
ax.set_ylabel('Average Duration(in minutes)')
plt.show()


# In[32]:


#Outliers of Duration


# In[33]:


Q1 = df['Duration'].quantile(0.25)
Q3 = df['Duration'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Duration'] >= lower_bound) & (df['Duration'] <= upper_bound)]
df.head(5)


# In[34]:


ax = seab.boxplot(data=df, y='Rating')
ax.set_ylabel('Rating')
ax.set_title('Box Plot of Movie Ratings')
plt.show()


# In[35]:


#Outliers of Rating


# In[36]:


Q1 = df['Rating'].quantile(0.25)
Q3 = df['Rating'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Rating'] >= lower_bound) & (df['Rating'] <= upper_bound)]
df.head(5)


# In[37]:


rating_votes = df.groupby('Rating')['Votes'].sum().reset_index()
plt.figure(figsize=(10, 6))
ax_line_seaborn = seab.lineplot(data=rating_votes, x='Rating', y='Votes', marker='o')
ax_line_seaborn.set_xlabel('Rating')
ax_line_seaborn.set_ylabel('Total Votes')
ax_line_seaborn.set_title('Total Votes per Rating')
plt.show()


# In[38]:


plt.figure(figsize=(10, 6))
ax = seab.barplot(x=directors.head(20).index, y=directors.head(20).values, palette='viridis')
ax.set_xlabel('Directors')
ax.set_ylabel('Frequency of Movies')
ax.set_title('Top 20 Directors by Frequency of Movies')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()


# In[39]:


plt.figure(figsize=(10, 6))
ax = seab.barplot(x=actors.head(20).index, y=actors.head(20).values, palette='viridis')
ax.set_xlabel('Actors')
ax.set_ylabel('Total Number of Movies')
ax.set_title('Top 20 Actors with Total Number of Movies')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()


# In[40]:


df["Actor"] = df['Actor 1'] + ', ' + df['Actor 2'] + ', ' + df['Actor 3']
df["Directors"] = df['Director'].astype('category').cat.codes
df["Genres"] = df['Genre'].astype('category').cat.codes
df["Actors"] = df['Actor'].astype('category').cat.codes
df.head(5)


# In[41]:


Q1 = df['Genres'].quantile(0.25)
Q3 = df['Genres'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Genres'] >= lower_bound) & (df['Genres'] <= upper_bound)]
df.head(5)


# In[42]:


ax = seab.boxplot(data=df, y='Directors')
ax.set_ylabel('Directors')
ax.set_title('Box Plot of Directors')
plt.show()


# In[43]:


Q1 = df['Directors'].quantile(0.25)
Q3 = df['Directors'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Directors'] >= lower_bound) & (df['Directors'] <= upper_bound)]
df.head(5)


# In[44]:


ax = seab.boxplot(data=df, y='Actors')
ax.set_ylabel('Actors')
ax.set_title('Box Plot of Actors')
plt.show()


# In[45]:


Q1 = df['Actors'].quantile(0.25)
Q3 = df['Actors'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Actors'] >= lower_bound) & (df['Actors'] <= upper_bound)]
df.head(5)


# In[46]:


ax = seab.histplot(data = df, x = "Rating", bins = 20, kde = True)
ax.set_xlabel('Rating')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Movie Ratings')
plt.show()


# In[47]:


#SPLITTING THE DATA


# In[48]:


from sklearn.model_selection import train_test_split


# In[49]:


Input = df.drop(['Name', 'Genre', 'Rating', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Actor'], axis=1)
Output = df['Rating']


# In[50]:


Input.head(5)


# In[51]:


x_train, x_test, y_train, y_test = train_test_split(Input, Output, test_size = 0.2, random_state = 1)


# In[52]:


#The Model


# In[53]:


def evaluate_model(y_true, y_pred, model_name):
    print("Model: ", model_name)
    print("Accuracy = {:0.2f}%".format(score(y_true, y_pred)*100))
    print("Mean Squared Error = {:0.2f}\n".format(mean_squared_error(y_true, y_pred, squared=False)))
    return round(score(y_true, y_pred)*100, 2)


# In[54]:


#Create a LinearRegression instance
LR = LinearRegression()

# Fit the model to your training data
LR.fit(x_train, y_train)

# Make predictions on the test data
lr_preds = LR.predict(x_test)
LRScore = evaluate_model(y_test, lr_preds, "LINEAR REGRESSION")


# In[55]:


DTR = DecisionTreeRegressor(random_state=1)
DTR.fit(x_train, y_train)
dt_preds = DTR.predict(x_test)
DTScore = evaluate_model(y_test, dt_preds, "DECEISION TREE")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




