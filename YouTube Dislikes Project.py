#!/usr/bin/env python
# coding: utf-8

# ## EDA on YouTube dislikes data using pandas
# ## This dataset contains information about trending YouTube videos from August 2020 to December 2021 for the USA, Canada, and Great Britain.

# ## Let's Analyze

# ## Q1. Import required libraries and read the provided dataset (youtube_dislike_dataset.csv) and retrieve top 5 and bottom 5 records.

# In[1]:


# Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Import the dataset (CSV file)

df = pd.read_csv(r"C:\Users\USER\Desktop\DA Projects\Python Project\youtube_dislike_dataset.csv")


# In[3]:


# First five records

df.head(5)


# In[4]:


# Last 5 records

df.tail(5)


# ## Q2. Check the info of the dataframe and write your inferences on data types and shape of the dataset.

# In[5]:


df.info()


# In[6]:


df.shape

The dataset has a total of 37422 entries ranging from index position 0 to 37421.
The dataset is spread across 12 columns.
The dataset has a shape of (37422,12) which means 37422 rows and 12 columns as said earlier.
It contains mostly object type data whereas the numerical data is of integer type. The only change to be made is to change the datatype of 'published_at' to datetime for convenience.
#  

# ## Q3. Check for the percentage of the missing values and drop or impute them.

# In[7]:


# Missing values percent of each colums w.r.t to complete dataset

null_percent = (df.isnull().sum()/len(df))*100
null_percent.round(2)

Only 'comments' column has missing values and only 0.42% are missing.
Since it is very negligible we can drop them and since the 'comments' column are opinion of people and it might not be possible to read all comments, we can drop the column
# In[8]:


df.drop(columns = 'comments', inplace= True)


# In[9]:


# 'comments' column has been dropped since it seems insignificant to our analysis
df.columns


# ## Q4. Check the statistical summary of both numerical and categorical columns and write your inferences.

# In[10]:


# describe() to see the statistical summary

df.describe() # By default, it displays only descriptive statistics of numeric columns


# In[11]:


# To include categorical data, we need to include 'all'

df.describe(include='all')

Maximum views for a video is 1.322797e+09.
Maximum Likes for a video is 3.183768e+07.
Maximum Dislikes for a video is 2.397733e+06.
Average view for a video is 5.697838e+06.
Maximum comments on a video is 1.607103e+07.
There are 10961 unique channel IDs.
#  ## Q5. Convert datatype of column published_at from object to pandas datetime.

# In[12]:


# Let us import datetime library
from datetime import datetime


# In[13]:


df['published_at'] = pd.to_datetime(df['published_at'])


# In[14]:


# Now let's check the new datatype of 'published_at'
df.dtypes


# ## Q6.  Create a new column as 'published_month' using the column published_at (display the months only)

# In[15]:


df['published_month'] = df['published_at'].dt.month
df.head()             
# dt.month to extract month from data. The new column is added last by default


# In[16]:


df[['published_month']]


# ## Q7.  Replace the numbers in the column published_month as names of the months i,e., 1 as 'Jan', 2 as 'Feb' and so on.....

# In[17]:


# We will use the map function to replace the value of months from numbers to their respective month names.

months = {1:'Jan', 2 :'Feb', 3 :'Mar',4: 'Apr',5:'May',6:'June',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
df['published_month'] = df['published_month'].map(lambda x : months[x])
df


# ## Q8.Find the number of videos published each month and arrange the months in a decreasing order based on the video count.

# In[18]:


# First method - value_counts()
df['published_month'].value_counts()


# In[19]:


# Second method - groupby()

df.groupby('published_month').count()['published_at'].sort_values(ascending=False)


# ## Q9.  Find the count of unique video_id, channel_id and channel_title.

# In[20]:


# nunique() will give unique value count of a column
df[['video_id','channel_id','channel_title']].nunique()


# ## Q10. Find the top10 channel names having the highest number of videos in the dataset and the bottom10 having lowest number of videos. 

# ### Using value_counts()

# In[21]:


df['channel_title'].value_counts().head(10)


# In[22]:


df['channel_title'].value_counts().tail(10)


# ### Using groupby()

# In[23]:


df.groupby(['channel_title']).count()['video_id'].sort_values(ascending=False).head(10)


# In[24]:


df.groupby(['channel_title']).count()['video_id'].sort_values(ascending=False).tail(10)

Note that value_counts() and groupby() sorts the values in a different way.
# ## Q11. Find the title of the video which has the maximum number of likes and the title of the video having minimum likes and write your inferences.
# First let us solve it by using 'groupby()' method
# In[25]:


most_liked = df.groupby(['title'])['likes'].sum().sort_values(ascending = False).head(1).reset_index()
most_liked


# In[26]:


least_liked = df.groupby(['title'])['likes'].sum().sort_values(ascending = False).tail(1).reset_index()
least_liked


#  

#  
# Now let us solve this by extracting the two columns and sorting them on the basis of 'likes'
# In[27]:


df[['title','likes']].sort_values(by= 'likes',ascending=False).head(1).reset_index(drop=True)


# In[28]:


df[['title','likes']].sort_values(by= 'likes',ascending=False).tail(1).reset_index(drop=True)


#  
BTS () 'Dynamite' Official MV has the most number of likes = 31837675'Kim Kardashian\'s Must-See Moments on "Saturday Night Live" has the least number of likes = 0
# ## Q12. Find the title of the video which has the maximum number of dislikes and the title of the video having minimum dislikes and write your inferences

# In[29]:


most_disliked = df.groupby('title')['dislikes'].max().sort_values(ascending = False).head(1).reset_index()
most_disliked

Cuties | Official Trailer | Netflix has the most number of dislikes = 2397733
# In[30]:


least_disliked = df.groupby('title')['dislikes'].max().sort_values(ascending = False).tail(1).reset_index()
least_disliked

'Kim Kardashian\'s Must-See Moments on "Saturday Night Live" has the least number of dislikes = 0
# ## Q13.  Does the number of views have any effect on how many people disliked the video? Support your answer with a metric and a plot

# In[31]:


sns.scatterplot(x = 'view_count', y = 'dislikes', data = df )
plt.show()


# In[32]:


sns.heatmap(df.corr(numeric_only= True), annot = True, cmap = 'viridis')
plt.show()

The scatter plot of view_count vs dislikes shows a proportional relationship between the two.
The correlation heatmap confirms it as it shows 0.68 as the correlation coefficient between view_count and dislikes.
0.68 says that the view_count has a significance effect on the number of dislikes.
# ## 14. Display all the information about the videos that were published in January, and mention the count of videos that were published in January.

# In[33]:


df[df['published_month'] == 'Jan']


# In[34]:


len(df[df['published_month'] == 'Jan'])


#  

# # End of Project
# # Thank You!
