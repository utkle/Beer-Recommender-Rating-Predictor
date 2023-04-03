# Beer Recommender & Rating Predictor

* Student name: A. Utku Kale
* Student pace: NYC/Full time
* Scheduled project review date/time: 4/1/23
* Instructor name: Brendan Hutchinson & Joseph Mata
* Blog post URL: https://medium.com/@utkukale 

![image.png](image/img2.jpg)

## Project Overwiev

This machine learning project is designed to help Rum & Lemonade Bistro to recommend other beers to their customers based on their preferance and help  brew their in house beer in the best way possible. The project aims to identify the most important features of beers that can help with rating prediction and popularity.

## Data

The dataset used for this machine learning project is a Beer Information - Tasting Profiles dataset obtained from Kaggle, up to 50 top-rated beers across 112 styles, 5558 beers in total. Source: BeerAdvocate.com. 


The dataset is divided into two parts. The first ten columns consist of information about the beer provided by the source, as well as additional data like a unique key for each beer and its style. The last eleven columns, on the other hand, represent the tasting profile features of the beer and are calculated based on the frequency of words used in up to 25 reviews for each beer. The assumption behind this is that people who write reviews are more likely to describe what they have experienced rather than what they have not.

The columns in the dataset are as follows:

Name: Beer's Name
\
\
Key: A unique key assigned to each beer
\
\
Style: Beer's Style
\
\
Style Key: A unique key assigned to each beer style
\
\
Brewery: Name of the beer's source
\
\
Description: Notes on the beer if available
\
\
Ave Rating: The average rating of the beer at the time of collection
\
\
Min IBU: The minimum International Bitterness Units value each beer can possess. 
\
\
Max IBU: The maximum International Bitterness Units value each beer can possess.
\
\
(Mouthfeel)
\
Astringency \
Body \
Alcohol \
\
\
(Taste) \
Bitter \
Sweet \
Sour \
Salty \
\
\
(Flavor And Aroma) \
Fruits \
Hoppy \
Spices \
Malty 
\
\
\
\
This dataset can be used to train a machine learning model to predict the average rating of a beer based on its features. The features in the dataset can also be analyzed to identify the most important ones for beer recommendations and beer rating predictions for Rum & Lemonade Bistro.



## Project steps
The data modeling process involved an exploratory data analysis phase, followed by the creation of baseline models and more advanced models using pipelines and grid searches. The performance of these models was evaluated on a hold-out test set, and the best-performing model was identified for rating prediction. Metric for this project is RMSE.


```python
# Loading the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import ipywidgets as widgets
from IPython.display import display
from sklearn.metrics import mean_squared_error, r2_score
from joypy import joyplot
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
import xgboost as xgb
from PIL import Image
```

## Exploratory Data Analysis

In this chapter, I will perform a comprehensive EDA of the Beer Profiles dataset, including investigation for any duplicate or null values, summary statistics, and correlation analysis. I will explore the distribution, range, and variability of each feature, as well as their relationships with each other and the target variable "Ave Rating". The insights and observations from this EDA will help the upcoming steps in my project, such as selecting relevant features, preprocessing the data, and fine-tuning the model. This chapter will provide a comprehensive overview of the data, its characteristics, and its suitability for the genre classification task, as well as demonstrate the importance of EDA in the data science workflow.


```python
# Loading the dataset into a Pandas DataFrame from a CSV file
df = pd.read_csv('data/beer_data_set.csv')

# Printing the first five rows of the DataFrame to check if the data was loaded correctly
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>key</th>
      <th>Style</th>
      <th>Style Key</th>
      <th>Brewery</th>
      <th>Description</th>
      <th>ABV</th>
      <th>Ave Rating</th>
      <th>Min IBU</th>
      <th>Max IBU</th>
      <th>...</th>
      <th>Body</th>
      <th>Alcohol</th>
      <th>Bitter</th>
      <th>Sweet</th>
      <th>Sour</th>
      <th>Salty</th>
      <th>Fruits</th>
      <th>Hoppy</th>
      <th>Spices</th>
      <th>Malty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Amber</td>
      <td>251</td>
      <td>Altbier</td>
      <td>8</td>
      <td>Alaskan Brewing Co.</td>
      <td>Notes:Richly malty and long on the palate, wit...</td>
      <td>5.3</td>
      <td>3.65</td>
      <td>25</td>
      <td>50</td>
      <td>...</td>
      <td>32</td>
      <td>9</td>
      <td>47</td>
      <td>74</td>
      <td>33</td>
      <td>0</td>
      <td>33</td>
      <td>57</td>
      <td>8</td>
      <td>111</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Double Bag</td>
      <td>252</td>
      <td>Altbier</td>
      <td>8</td>
      <td>Long Trail Brewing Co.</td>
      <td>Notes:This malty, full-bodied double alt is al...</td>
      <td>7.2</td>
      <td>3.90</td>
      <td>25</td>
      <td>50</td>
      <td>...</td>
      <td>57</td>
      <td>18</td>
      <td>33</td>
      <td>55</td>
      <td>16</td>
      <td>0</td>
      <td>24</td>
      <td>35</td>
      <td>12</td>
      <td>84</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Long Trail Ale</td>
      <td>253</td>
      <td>Altbier</td>
      <td>8</td>
      <td>Long Trail Brewing Co.</td>
      <td>Notes:Long Trail Ale is a full-bodied amber al...</td>
      <td>5.0</td>
      <td>3.58</td>
      <td>25</td>
      <td>50</td>
      <td>...</td>
      <td>37</td>
      <td>6</td>
      <td>42</td>
      <td>43</td>
      <td>11</td>
      <td>0</td>
      <td>10</td>
      <td>54</td>
      <td>4</td>
      <td>62</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Doppelsticke</td>
      <td>254</td>
      <td>Altbier</td>
      <td>8</td>
      <td>Uerige Obergärige Hausbrauerei</td>
      <td>Notes:</td>
      <td>8.5</td>
      <td>4.15</td>
      <td>25</td>
      <td>50</td>
      <td>...</td>
      <td>55</td>
      <td>31</td>
      <td>47</td>
      <td>101</td>
      <td>18</td>
      <td>1</td>
      <td>49</td>
      <td>40</td>
      <td>16</td>
      <td>119</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Scurry</td>
      <td>255</td>
      <td>Altbier</td>
      <td>8</td>
      <td>Off Color Brewing</td>
      <td>Notes:Just cause it's dark and German doesn't ...</td>
      <td>5.3</td>
      <td>3.67</td>
      <td>25</td>
      <td>50</td>
      <td>...</td>
      <td>69</td>
      <td>10</td>
      <td>63</td>
      <td>120</td>
      <td>14</td>
      <td>0</td>
      <td>19</td>
      <td>36</td>
      <td>15</td>
      <td>218</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
# Printing the shape of the dataset.
print(f'The dataset has {df.shape[0]} rows and {df.shape[1]} columns.')

# Printing descriptive statistics of the numerical columns in the DataFrame, such as count, mean, std, min, max, and quartiles
df.describe().T
```

    The dataset has 5558 rows and 21 columns.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>key</th>
      <td>5558.0</td>
      <td>2779.500000</td>
      <td>1604.600729</td>
      <td>1.00</td>
      <td>1390.25</td>
      <td>2779.50</td>
      <td>4168.75</td>
      <td>5558.00</td>
    </tr>
    <tr>
      <th>Style Key</th>
      <td>5558.0</td>
      <td>64.449082</td>
      <td>35.814930</td>
      <td>2.00</td>
      <td>34.00</td>
      <td>64.00</td>
      <td>95.00</td>
      <td>126.00</td>
    </tr>
    <tr>
      <th>ABV</th>
      <td>5558.0</td>
      <td>6.633730</td>
      <td>2.521660</td>
      <td>0.00</td>
      <td>5.00</td>
      <td>6.00</td>
      <td>7.90</td>
      <td>57.50</td>
    </tr>
    <tr>
      <th>Ave Rating</th>
      <td>5558.0</td>
      <td>3.760239</td>
      <td>0.442951</td>
      <td>1.27</td>
      <td>3.59</td>
      <td>3.82</td>
      <td>4.04</td>
      <td>4.83</td>
    </tr>
    <tr>
      <th>Min IBU</th>
      <td>5558.0</td>
      <td>20.715545</td>
      <td>13.736873</td>
      <td>0.00</td>
      <td>10.00</td>
      <td>20.00</td>
      <td>25.00</td>
      <td>65.00</td>
    </tr>
    <tr>
      <th>Max IBU</th>
      <td>5558.0</td>
      <td>38.452321</td>
      <td>22.184524</td>
      <td>0.00</td>
      <td>25.00</td>
      <td>35.00</td>
      <td>45.00</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Astringency</th>
      <td>5558.0</td>
      <td>15.938647</td>
      <td>11.624254</td>
      <td>0.00</td>
      <td>8.00</td>
      <td>14.00</td>
      <td>22.00</td>
      <td>83.00</td>
    </tr>
    <tr>
      <th>Body</th>
      <td>5558.0</td>
      <td>42.746132</td>
      <td>28.589959</td>
      <td>0.00</td>
      <td>25.00</td>
      <td>38.00</td>
      <td>55.00</td>
      <td>197.00</td>
    </tr>
    <tr>
      <th>Alcohol</th>
      <td>5558.0</td>
      <td>15.975171</td>
      <td>18.268342</td>
      <td>0.00</td>
      <td>5.00</td>
      <td>10.00</td>
      <td>20.00</td>
      <td>139.00</td>
    </tr>
    <tr>
      <th>Bitter</th>
      <td>5558.0</td>
      <td>34.316121</td>
      <td>27.118100</td>
      <td>0.00</td>
      <td>13.00</td>
      <td>29.00</td>
      <td>51.00</td>
      <td>150.00</td>
    </tr>
    <tr>
      <th>Sweet</th>
      <td>5558.0</td>
      <td>53.629723</td>
      <td>35.866101</td>
      <td>0.00</td>
      <td>27.00</td>
      <td>49.50</td>
      <td>74.00</td>
      <td>263.00</td>
    </tr>
    <tr>
      <th>Sour</th>
      <td>5558.0</td>
      <td>34.610291</td>
      <td>39.850228</td>
      <td>0.00</td>
      <td>9.00</td>
      <td>21.00</td>
      <td>44.00</td>
      <td>323.00</td>
    </tr>
    <tr>
      <th>Salty</th>
      <td>5558.0</td>
      <td>1.314142</td>
      <td>3.874110</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>66.00</td>
    </tr>
    <tr>
      <th>Fruits</th>
      <td>5558.0</td>
      <td>39.378553</td>
      <td>36.652293</td>
      <td>0.00</td>
      <td>10.00</td>
      <td>28.00</td>
      <td>61.75</td>
      <td>222.00</td>
    </tr>
    <tr>
      <th>Hoppy</th>
      <td>5558.0</td>
      <td>38.414538</td>
      <td>31.912843</td>
      <td>0.00</td>
      <td>14.00</td>
      <td>30.00</td>
      <td>56.00</td>
      <td>193.00</td>
    </tr>
    <tr>
      <th>Spices</th>
      <td>5558.0</td>
      <td>17.584023</td>
      <td>23.973879</td>
      <td>0.00</td>
      <td>4.00</td>
      <td>9.00</td>
      <td>22.00</td>
      <td>184.00</td>
    </tr>
    <tr>
      <th>Malty</th>
      <td>5558.0</td>
      <td>68.591400</td>
      <td>44.600385</td>
      <td>0.00</td>
      <td>33.00</td>
      <td>65.00</td>
      <td>99.00</td>
      <td>304.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Printing information about the DataFrame, including the data types of each column and the number of non-null values
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5558 entries, 0 to 5557
    Data columns (total 21 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   Name         5556 non-null   object 
     1   key          5558 non-null   int64  
     2   Style        5558 non-null   object 
     3   Style Key    5558 non-null   int64  
     4   Brewery      5558 non-null   object 
     5   Description  5558 non-null   object 
     6   ABV          5558 non-null   float64
     7   Ave Rating   5558 non-null   float64
     8   Min IBU      5558 non-null   int64  
     9   Max IBU      5558 non-null   int64  
     10  Astringency  5558 non-null   int64  
     11  Body         5558 non-null   int64  
     12  Alcohol      5558 non-null   int64  
     13  Bitter       5558 non-null   int64  
     14  Sweet        5558 non-null   int64  
     15  Sour         5558 non-null   int64  
     16  Salty        5558 non-null   int64  
     17  Fruits       5558 non-null   int64  
     18  Hoppy        5558 non-null   int64  
     19  Spices       5558 non-null   int64  
     20  Malty        5558 non-null   int64  
    dtypes: float64(2), int64(15), object(4)
    memory usage: 912.0+ KB



```python
# Checking for missing values in each column of the DataFrame and print the total number of missing values
df.isna().sum()
```




    Name           2
    key            0
    Style          0
    Style Key      0
    Brewery        0
    Description    0
    ABV            0
    Ave Rating     0
    Min IBU        0
    Max IBU        0
    Astringency    0
    Body           0
    Alcohol        0
    Bitter         0
    Sweet          0
    Sour           0
    Salty          0
    Fruits         0
    Hoppy          0
    Spices         0
    Malty          0
    dtype: int64




```python
df[df['Name'].isna()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>key</th>
      <th>Style</th>
      <th>Style Key</th>
      <th>Brewery</th>
      <th>Description</th>
      <th>ABV</th>
      <th>Ave Rating</th>
      <th>Min IBU</th>
      <th>Max IBU</th>
      <th>...</th>
      <th>Body</th>
      <th>Alcohol</th>
      <th>Bitter</th>
      <th>Sweet</th>
      <th>Sour</th>
      <th>Salty</th>
      <th>Fruits</th>
      <th>Hoppy</th>
      <th>Spices</th>
      <th>Malty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1803</th>
      <td>NaN</td>
      <td>3504</td>
      <td>Kvass</td>
      <td>81</td>
      <td>Monastyrskiy Kvas</td>
      <td>Notes:</td>
      <td>1.5</td>
      <td>3.07</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>34</td>
      <td>4</td>
      <td>15</td>
      <td>84</td>
      <td>16</td>
      <td>1</td>
      <td>33</td>
      <td>14</td>
      <td>7</td>
      <td>68</td>
    </tr>
    <tr>
      <th>2150</th>
      <td>NaN</td>
      <td>2401</td>
      <td>Lager - European Pale</td>
      <td>57</td>
      <td>Stella Artois</td>
      <td>Notes:</td>
      <td>5.0</td>
      <td>3.11</td>
      <td>18</td>
      <td>25</td>
      <td>...</td>
      <td>14</td>
      <td>10</td>
      <td>20</td>
      <td>19</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>26</td>
      <td>3</td>
      <td>26</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 21 columns</p>
</div>



2 missing values in Name column, I can fill the name column manually.


```python
df.loc[1803,'Name'] = 'Monastyrskiy Kvas'
df.loc[2150,'Name'] = 'Stella Artois'
```


```python
# Checking for duplicated rows in the DataFrame and print the total number of duplicates
df.duplicated().sum()
```




    0




```python
# Finding the top-rated beers based on the review_overall column
top_rated = df[['Name', 'Style', 'Brewery','ABV','Ave Rating']].sort_values('Ave Rating', ascending=False).reset_index().head(10)
top_rated
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Name</th>
      <th>Style</th>
      <th>Brewery</th>
      <th>ABV</th>
      <th>Ave Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>499</td>
      <td>eisbock</td>
      <td>Bock - Eisbock</td>
      <td>Kulmbacher Kommunbräu</td>
      <td>9.20</td>
      <td>4.83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4779</td>
      <td>Marshmallow Handjee</td>
      <td>Stout - Russian Imperial</td>
      <td>3 Floyds Brewing Co.</td>
      <td>15.00</td>
      <td>4.81</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1750</td>
      <td>Heady Topper</td>
      <td>IPA - New England</td>
      <td>The Alchemist</td>
      <td>8.00</td>
      <td>4.76</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1777</td>
      <td>King Julius</td>
      <td>IPA - New England</td>
      <td>Tree House Brewing Company</td>
      <td>8.30</td>
      <td>4.76</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2919</td>
      <td>Zenne Y Frontera</td>
      <td>Lambic - Traditional</td>
      <td>Brouwerij 3 Fonteinen</td>
      <td>7.00</td>
      <td>4.75</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1793</td>
      <td>Very Hazy</td>
      <td>IPA - New England</td>
      <td>Tree House Brewing Company</td>
      <td>8.60</td>
      <td>4.75</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1727</td>
      <td>Pliny The Younger</td>
      <td>IPA - Imperial</td>
      <td>Russian River Brewing Company</td>
      <td>10.25</td>
      <td>4.75</td>
    </tr>
    <tr>
      <th>7</th>
      <td>284</td>
      <td>Drone Witch</td>
      <td>Bière de Champagne / Bière Brut</td>
      <td>Heirloom Rustic Ales</td>
      <td>6.60</td>
      <td>4.74</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4463</td>
      <td>Bourbon County Brand Coffee Stout</td>
      <td>Stout - American Imperial</td>
      <td>Goose Island Beer Co.</td>
      <td>12.90</td>
      <td>4.73</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1784</td>
      <td>Juice Machine</td>
      <td>IPA - New England</td>
      <td>Tree House Brewing Company</td>
      <td>8.20</td>
      <td>4.71</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Creating a Bar plot to visualize top rated beers
sns.barplot(top_rated, x='Ave Rating', y='Name')
plt.title('Top-Rated Beers', fontsize=20)
plt.xlabel('Overall Rating', fontsize=16)
plt.ylabel('Beer Name', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False, axis='y')
plt.grid(axis='x', linestyle='--', color='grey', alpha=0.5)
plt.ylim(-0.5, 9.5)
plt.xlim(4.25, 5.0)
```




    (4.25, 5.0)




    
![png](output_17_1.png)
    



```python
# Finding the lowest-rated beers based on the review_overall column
lowest_rated = df[['Name', 'Style', 'Brewery','ABV','Ave Rating']].sort_values('Ave Rating', ascending=False).reset_index().tail(10)
lowest_rated
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Name</th>
      <th>Style</th>
      <th>Brewery</th>
      <th>ABV</th>
      <th>Ave Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5548</th>
      <td>1918</td>
      <td>Natural Ice</td>
      <td>Lager - Adjunct</td>
      <td>Anheuser-Busch</td>
      <td>5.9</td>
      <td>1.70</td>
    </tr>
    <tr>
      <th>5549</th>
      <td>1930</td>
      <td>Keystone IceCoors Brewing Company (Molson-Coors)</td>
      <td>Lager - Adjunct</td>
      <td>Coors Brewing Company (Molson-Coors)</td>
      <td>5.9</td>
      <td>1.65</td>
    </tr>
    <tr>
      <th>5550</th>
      <td>1375</td>
      <td>FiLite</td>
      <td>Happoshu</td>
      <td>Hite Brewery Company LTD</td>
      <td>4.5</td>
      <td>1.64</td>
    </tr>
    <tr>
      <th>5551</th>
      <td>992</td>
      <td>El Lapino</td>
      <td>Chile Beer</td>
      <td>Microbrasserie du Lièvre</td>
      <td>5.4</td>
      <td>1.63</td>
    </tr>
    <tr>
      <th>5552</th>
      <td>2456</td>
      <td>Natural Light</td>
      <td>Lager - Light</td>
      <td>Anheuser-Busch</td>
      <td>4.2</td>
      <td>1.58</td>
    </tr>
    <tr>
      <th>5553</th>
      <td>2967</td>
      <td>Sharp's</td>
      <td>Low Alcohol Beer</td>
      <td>Miller Brewing Co.</td>
      <td>0.4</td>
      <td>1.57</td>
    </tr>
    <tr>
      <th>5554</th>
      <td>2471</td>
      <td>Budweiser Select 55</td>
      <td>Lager - Light</td>
      <td>Anheuser-Busch</td>
      <td>2.4</td>
      <td>1.53</td>
    </tr>
    <tr>
      <th>5555</th>
      <td>2478</td>
      <td>Miller Genuine Draft 64</td>
      <td>Lager - Light</td>
      <td>Miller Brewing Co.</td>
      <td>3.0</td>
      <td>1.50</td>
    </tr>
    <tr>
      <th>5556</th>
      <td>2533</td>
      <td>Evil Eye</td>
      <td>Lager - Malt Liquor</td>
      <td>Melanie Brewing Company</td>
      <td>10.0</td>
      <td>1.48</td>
    </tr>
    <tr>
      <th>5557</th>
      <td>2399</td>
      <td>Siamsato</td>
      <td>Lager - Japanese Rice</td>
      <td>Siamsato Brewery</td>
      <td>8.0</td>
      <td>1.27</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Creating a Bar plot to visualize lowest rated beers
sns.barplot(lowest_rated, x='Ave Rating', y='Name')
plt.title('Lowest-Rated Beers', fontsize=20)
plt.xlabel('Overall Rating', fontsize=16)
plt.ylabel('Beer Name', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False, axis='y')
plt.grid(axis='x', linestyle='--', color='grey', alpha=0.5)
plt.ylim(-0.5, 9.5)
plt.xlim(0, 3.0)
```




    (0.0, 3.0)




    
![png](output_19_1.png)
    



```python
# Finding the top rated breweries based on the review_overall column
top_rated_breweries = df[['Name', 'Style', 'Brewery','ABV','Ave Rating']].groupby('Style').mean().sort_values('Ave Rating')
top_rated_breweries
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ABV</th>
      <th>Ave Rating</th>
    </tr>
    <tr>
      <th>Style</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Lager - Malt Liquor</th>
      <td>7.639600</td>
      <td>2.35920</td>
    </tr>
    <tr>
      <th>Lager - Light</th>
      <td>3.978800</td>
      <td>2.40400</td>
    </tr>
    <tr>
      <th>Lager - Adjunct</th>
      <td>4.815400</td>
      <td>2.62860</td>
    </tr>
    <tr>
      <th>Low Alcohol Beer</th>
      <td>0.442600</td>
      <td>2.85360</td>
    </tr>
    <tr>
      <th>Lager - European Strong</th>
      <td>8.718000</td>
      <td>2.94380</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Stout - Russian Imperial</th>
      <td>11.049000</td>
      <td>4.23320</td>
    </tr>
    <tr>
      <th>IPA - Imperial</th>
      <td>9.414000</td>
      <td>4.27200</td>
    </tr>
    <tr>
      <th>Wild Ale</th>
      <td>7.668163</td>
      <td>4.30551</td>
    </tr>
    <tr>
      <th>Stout - American Imperial</th>
      <td>11.662000</td>
      <td>4.36780</td>
    </tr>
    <tr>
      <th>IPA - New England</th>
      <td>7.780000</td>
      <td>4.50500</td>
    </tr>
  </tbody>
</table>
<p>112 rows × 2 columns</p>
</div>



Highest rated beer is Bock eisbock with 4.83 and lowest rated beer is Japanese Rice Lager Siamsoto with 1.27. Top three highest rated beer styles are IPA - New England, Stout - American Imperial and Wild Ale. 


```python
# I will create a dataframe for numeric columns to observe their relationship.
df_numeric = df[['ABV', 'Ave Rating','Min IBU', 'Max IBU', 'Astringency', 'Body', 'Alcohol', 'Bitter', 'Sweet', 'Sour', 'Salty', 'Fruits', 'Hoppy', 'Spices','Malty']]
```


```python
# Using MinMaxScaler to scale numeric values.
scaler = MinMaxScaler(feature_range=(-1,1))
df2 = scaler.fit_transform(df_numeric)
df_num_sc = pd.DataFrame(df2, index=df_numeric.index, columns=df_numeric.columns)
df_num_sc.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ABV</th>
      <td>5558.0</td>
      <td>-0.769262</td>
      <td>0.087710</td>
      <td>-1.0</td>
      <td>-0.826087</td>
      <td>-0.791304</td>
      <td>-0.725217</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Ave Rating</th>
      <td>5558.0</td>
      <td>0.399011</td>
      <td>0.248849</td>
      <td>-1.0</td>
      <td>0.303371</td>
      <td>0.432584</td>
      <td>0.556180</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Min IBU</th>
      <td>5558.0</td>
      <td>-0.362599</td>
      <td>0.422673</td>
      <td>-1.0</td>
      <td>-0.692308</td>
      <td>-0.384615</td>
      <td>-0.230769</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Max IBU</th>
      <td>5558.0</td>
      <td>-0.230954</td>
      <td>0.443690</td>
      <td>-1.0</td>
      <td>-0.500000</td>
      <td>-0.300000</td>
      <td>-0.100000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Astringency</th>
      <td>5558.0</td>
      <td>-0.615936</td>
      <td>0.280103</td>
      <td>-1.0</td>
      <td>-0.807229</td>
      <td>-0.662651</td>
      <td>-0.469880</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Body</th>
      <td>5558.0</td>
      <td>-0.566029</td>
      <td>0.290253</td>
      <td>-1.0</td>
      <td>-0.746193</td>
      <td>-0.614213</td>
      <td>-0.441624</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Alcohol</th>
      <td>5558.0</td>
      <td>-0.770141</td>
      <td>0.262854</td>
      <td>-1.0</td>
      <td>-0.928058</td>
      <td>-0.856115</td>
      <td>-0.712230</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Bitter</th>
      <td>5558.0</td>
      <td>-0.542452</td>
      <td>0.361575</td>
      <td>-1.0</td>
      <td>-0.826667</td>
      <td>-0.613333</td>
      <td>-0.320000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Sweet</th>
      <td>5558.0</td>
      <td>-0.592169</td>
      <td>0.272746</td>
      <td>-1.0</td>
      <td>-0.794677</td>
      <td>-0.623574</td>
      <td>-0.437262</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Sour</th>
      <td>5558.0</td>
      <td>-0.785695</td>
      <td>0.246751</td>
      <td>-1.0</td>
      <td>-0.944272</td>
      <td>-0.869969</td>
      <td>-0.727554</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Salty</th>
      <td>5558.0</td>
      <td>-0.960178</td>
      <td>0.117397</td>
      <td>-1.0</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-0.969697</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Fruits</th>
      <td>5558.0</td>
      <td>-0.645238</td>
      <td>0.330201</td>
      <td>-1.0</td>
      <td>-0.909910</td>
      <td>-0.747748</td>
      <td>-0.443694</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Hoppy</th>
      <td>5558.0</td>
      <td>-0.601922</td>
      <td>0.330703</td>
      <td>-1.0</td>
      <td>-0.854922</td>
      <td>-0.689119</td>
      <td>-0.419689</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Spices</th>
      <td>5558.0</td>
      <td>-0.808869</td>
      <td>0.260586</td>
      <td>-1.0</td>
      <td>-0.956522</td>
      <td>-0.902174</td>
      <td>-0.760870</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Malty</th>
      <td>5558.0</td>
      <td>-0.548741</td>
      <td>0.293424</td>
      <td>-1.0</td>
      <td>-0.782895</td>
      <td>-0.572368</td>
      <td>-0.348684</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



After scaling my numeric dataset, I will divide each feature into 5 intervals. This will help me to observe and interpret visualizations easier. 


```python
# Define the categories and labels for each column
categories = {
    'ABV': ['1','2','3','4','5'],
    'Ave Rating': ['1','2','3','4','5'],
    'Min IBU': ['1','2','3','4','5'],
    'Max IBU': ['1','2','3','4','5'],
    'Astringency': ['1','2','3','4','5'],
    'Body': ['1','2','3','4','5'],
    'Alcohol': ['1','2','3','4','5'],
    'Bitter': ['1','2','3','4','5'],
    'Sweet': ['1','2','3','4','5'],
    'Sour': ['1','2','3','4','5'],
    'Salty': ['1','2','3','4','5'],
    'Fruits': ['1','2','3','4','5'],
    'Hoppy': ['1','2','3','4','5'],
    'Spices': ['1','2','3','4','5'],
    'Malty': ['1','2','3','4','5']
}

# Loop through each column and create categories using the cut function
for col in categories.keys():
    df_num_sc[col] = pd.cut(df_num_sc[col], bins=5, labels=categories[col])

# Verify that the column values have been replaced with category names

```


```python
# Creating the heatmap to check for multicollinearity and correlation of Ave Rating and features.
plt.figure(figsize=(16,16))
sns.heatmap(df_numeric.corr(), annot = True)
plt.show()
```


    
![png](output_26_0.png)
    



```python
# Creating boxplots to check correlation of Ave Rating and features.
plt.figure(figsize=(16,16))
for i, col in enumerate(df_numeric.columns):
    plt.subplot(4,4,i+1)
    sns.boxplot(y=df_numeric[col],x=df_num_sc['Ave Rating'] )
    sns.set_style("darkgrid")
plt.tight_layout()

plt.show()
```


    
![png](output_27_0.png)
    



```python
# Creating scatter plots, histograms and distribution plots to visualize distribution of features and target.
df_numeric.hist(figsize=(16,16))
plt.tight_layout()
plt.plot()
```




    []




    
![png](output_28_1.png)
    



```python
plt.figure(figsize=(16,16))
for i, col in enumerate(df_numeric.columns):
    plt.subplot(4,4,i+1)
    sns.distplot(df_numeric[col])
    sns.set_style("darkgrid")
plt.tight_layout()

plt.show()


```


    
![png](output_29_0.png)
    



```python
plt.figure(figsize=(16,16))
for i, col in enumerate(df_numeric.columns):
    plt.subplot(4,4,i+1)
    sns.regplot(df_numeric, x='Ave Rating', y=col, line_kws={"color": "black"}, scatter_kws={'s':2})
    sns.set_style("darkgrid")
plt.tight_layout()

plt.show()
```


    
![png](output_30_0.png)
    


From the EDA, there is no strong correlation between features and target. Target 'Ave Rating' is mainly distributed between 3 and 4.5 rating. Some of the features such as 'Astringency', 'Body', 'Alcohol', 'Bitter', 'Sour', 'Fruits', 'Hoppy', 'Spices' has skewness, I will use log transform to help transform their distribution to normal.For the categorical features 'Style' and 'Brewery' I will use encoder. 

## Data preprocessing & running the regression models

##### Data preprocessing with transformers


```python
# Dropping unnecessary columns
df_pp = df.drop(columns=['key', 'Style Key','Description','Name','Salty'], axis=1)
# Converting the 'Style' and 'Brewery' columns into categorical data
df_pp['Style'] = pd.Categorical(df['Style'])
df_pp['Brewery'] = pd.Categorical(df['Brewery'])
# Defining the numeric and categorical transformers for ColumnTransformer
num_transformer = ('scaler', StandardScaler(),  ['ABV', 'Min IBU', 'Max IBU', 'Astringency', 'Body', 'Alcohol', 'Bitter', 'Sweet', 'Sour', 'Fruits', 'Hoppy', 'Spices', 'Malty'])
cat_transformer = ('onehot', OneHotEncoder(handle_unknown='ignore'), ['Style','Brewery'])
log_transformer = ('log', FunctionTransformer(lambda x: np.log(x + 1e-6)),['Astringency', 'Body', 'Alcohol', 'Bitter', 'Sour', 'Fruits', 'Hoppy', 'Spices'] )
# Creating the ColumnTransformer preprocessor using the transformers
preprocessor = ColumnTransformer(transformers=[log_transformer,num_transformer, cat_transformer])

```

##### Train test split


```python
# I'm defining my target column y and predictor columns X
y = df_pp['Ave Rating']
X = df_pp.drop(columns='Ave Rating')
```


```python
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=10, test_size=0.2)
```

### 1st Model: Random Forest


```python
# Creating a pipeline that includes preprocessing and Random Forest Regressor
pipe_rf = Pipeline([('preprocessing',preprocessor),('rf',RandomForestRegressor())])

# Fitting the pipeline on the training data
pipe_rf.fit(X_train, y_train)

# Printing the mean validation score of the pipeline
print("Validation score: ", cross_val_score(pipe_rf,X_train,y_train).mean())

#Test data accuracy score 
y_pred_test_rf = pipe_rf.predict(X_test)
rftestscore = r2_score(y_test, y_pred_test_rf)
rftestscorermse = mean_squared_error(y_test, y_pred_test_rf, squared=False)
#Train data accuracy score
y_pred_train_rf = pipe_rf.predict(X_train)
rftrainscore = r2_score(y_train, y_pred_train_rf)
rftrainscorermse = mean_squared_error(y_train, y_pred_train_rf, squared=False)
print(f"Train Score: {rftrainscore}, Test Score: {rftestscore}")
print(f"Train RMSE: {rftrainscorermse}, Test RMSE: {rftestscorermse}")
```

    Validation score:  0.6617900419002088
    Train Score: 0.9551821822074585, Test Score: 0.7120772025322233
    Train RMSE: 0.0932614491226829, Test RMSE: 0.24267041041380905



```python
# Defining a hyperparameter grid for Random Forest
rf_grid = {
   'rf__n_estimators': [100,200,500,800],
    'rf__max_depth': [3,6,8,10] }
```


```python
# Defining a GridSearchCV object to search for the best hyperparameters for the pipeline with the hyperparamater grid
rf_gs = GridSearchCV(estimator=pipe_rf, param_grid=rf_grid,n_jobs=-1, cv=5)

# Fitting the GridSearchCV object on the training data
rf_model = rf_gs.fit(X_train,y_train)

# Printing validation score and best hyperparemeters
print("Validation score: ", rf_gs.best_score_)
print("Best hyperparameters: ", rf_gs.best_params_)

#Test data accuracy score 
y_pred_test_rf = rf_model.best_estimator_.predict(X_test)
rftestscore = r2_score(y_test, y_pred_test_rf)
rftestscorermse = mean_squared_error(y_test, y_pred_test_rf, squared=False)
#Train data accuracy score
y_pred_train_rf = rf_model.best_estimator_.predict(X_train)
rftrainscore = r2_score(y_train, y_pred_train_rf)
rftrainscorermse = mean_squared_error(y_train, y_pred_train_rf, squared=False)
print(f"Train R2: {rftrainscore}, Test R2: {rftestscore}")
print(f"Train RMSE: {rftrainscorermse}, Test RMSE: {rftestscorermse}")
```

    Validation score:  0.6155723668792664
    Best hyperparameters:  {'rf__max_depth': 10, 'rf__n_estimators': 100}
    Train R2: 0.7365718721581428, Test R2: 0.6618903884533445
    Train RMSE: 0.22610380905112318, Test RMSE: 0.2629708174425402


### 2nd Model: Linear Regression


```python
# Creating a pipeline that includes preprocessing and Linear Regression
pipe_lr = Pipeline([('preprocessing',preprocessor),('lr', LinearRegression())])

# Fitting the pipeline on the training data
pipe_lr.fit(X_train, y_train)

# Printing the mean validation score of the pipeline
print("Validation score: ", cross_val_score(pipe_lr,X_train,y_train).mean())

#Test data accuracy score 
y_pred_test_lr = pipe_lr.predict(X_test)
lrtestscore = r2_score(y_test, y_pred_test_lr)
lrtestscorermse = mean_squared_error(y_test, y_pred_test_lr,squared=False)
#Train data accuracy score
y_pred_train_lr = pipe_lr.predict(X_train)
lrtrainscore = r2_score(y_train, y_pred_train_lr)
lrtrainscorermse = mean_squared_error(y_train, y_pred_train_lr,squared=False)
print(f"Train R2: {lrtrainscore}, Test R2: {lrtestscore}")
print(f"Train RMSE: {lrtrainscorermse}, Test RMSE: {lrtestscorermse}")
```

    Validation score:  0.6964867137150689
    Train R2: 0.8867898938492323, Test R2: 0.7272459566827825
    Train RMSE: 0.14822429891057315, Test RMSE: 0.23619157198680998


I received a better score for this model. I will check feature importances to see which features affect target the most.


```python
importance = pipe_lr.named_steps['lr'].coef_
feature_names = pipe_lr.named_steps['preprocessing'].transformers_[0][2] + pipe_lr.named_steps['preprocessing'].transformers_[1][2] + pipe_lr.named_steps['preprocessing'].transformers_[2][2]
# Add the feature names to the importance scores
importance_with_names = zip(feature_names, importance)
# Sort the features by importance
sorted_importance = sorted(importance_with_names, key=lambda x: x[1], reverse=True)
# Plot feature importance with names
pyplot.bar([x[0] for x in sorted_importance], [x[1] for x in sorted_importance])
pyplot.xticks(rotation=90)
pyplot.show()
```


    
![png](output_45_0.png)
    


Best features are: ABV, Fruits, Bitter

### 3rd Model: Ridge


```python
# Creating a pipeline that includes preprocessing and Ridge
ridge_pipe = Pipeline([('preprocessing',preprocessor),('ridge',Ridge(normalize=True))])
ridge_pipe.fit(X_train, y_train)
print("Validation score: ", cross_val_score(ridge_pipe,X_train,y_train).mean())
#Test data accuracy score 
y_pred_test_ridge = ridge_pipe.predict(X_test)
ridgetestscore = r2_score(y_test, y_pred_test_ridge)
ridgetestscorermse = mean_squared_error(y_test, y_pred_test_ridge,squared=False)
#Train data accuracy score
y_pred_train_ridge = ridge_pipe.predict(X_train)
ridgetrainscore = r2_score(y_train, y_pred_train_ridge)
ridgetrainscorermse = mean_squared_error(y_train, y_pred_train_ridge,squared=False)
print(f"Train R2: {ridgetrainscore}, Test R2: {ridgetestscore}")
print(f"Train RMSE: {ridgetrainscorermse}, Test RMSE: {ridgetestscorermse}")
```

    Validation score:  0.6503520854787128
    Train R2: 0.7673081917482744, Test R2: 0.6768645645868009
    Train RMSE: 0.2125041194104468, Test RMSE: 0.2570816572771408



```python
# Defining a hyperparameter grid for Ridge
param_grid = {
    'ridge__alpha': [1,1.5,1.2,2,0.1],
    'ridge__normalize': [True, False],
    'ridge__solver': ['auto', 'saga', 'lbfgs']
    #'ridge__tol': [0.001,0.0001,0.01,0.1]
}
```


```python
# Defining a GridSearchCV object to search for the best hyperparameters for the pipeline with the hyperparamater grid
ridge_gs = GridSearchCV(
    ridge_pipe,
    param_grid,
    cv=5,
)

ridge_gs.fit(X_train, y_train)

print("Validation score: ", ridge_gs.best_score_)
print("Best hyperparameters: ", ridge_gs.best_params_)

#Test data accuracy score 
y_pred_test_ridge_gs = ridge_gs.best_estimator_.predict(X_test)
ridgetestscoregs = r2_score(y_test, y_pred_test_ridge_gs)
ridgetestscoregsrmse = mean_squared_error(y_test, y_pred_test_ridge_gs,squared=False)
#Train data accuracy score
y_pred_train_ridge_gs = ridge_gs.best_estimator_.predict(X_train)
ridgetrainscoregs = r2_score(y_train, y_pred_train_ridge_gs)
ridgetrainscoregsrmse = mean_squared_error(y_train, y_pred_train_ridge_gs,squared=False)
print(f"Train R2: {ridgetrainscoregs}, Test R2: {ridgetestscoregs}")
print(f"Train RMSE: {ridgetrainscoregsrmse}, Test RMSE: {ridgetestscoregsrmse}")
```

    Validation score:  0.7228542810919805
    Best hyperparameters:  {'ridge__alpha': 1, 'ridge__normalize': False, 'ridge__solver': 'auto'}
    Train R2: 0.8551531996023402, Test R2: 0.7287423104678263
    Train RMSE: 0.16766069387142382, Test RMSE: 0.23554279660157038


### 4th Model: XGB


```python
# Creating a pipeline that includes preprocessing and XG Boosting
pipe_xgb = Pipeline([('preprocessing',preprocessor),('xgb', xgb.XGBRegressor(verbosity = 0, silent=True))])

pipe_xgb.fit(X_train, y_train)

# Printing the mean validation score of the pipeline
print("Validation score: ", cross_val_score(pipe_xgb,X_train,y_train).mean())
#Test data accuracy score 
y_pred_test_xgb = pipe_xgb.predict(X_test)
xgbtestscore = r2_score(y_test, y_pred_test_xgb)
xgbtestscorermse = mean_squared_error(y_test, y_pred_test_xgb,squared=False)
#Train data accuracy score
y_pred_train_xgb = pipe_xgb.predict(X_train)
xgbtrainscore = r2_score(y_train, y_pred_train_xgb)
xgbtrainscorermse = mean_squared_error(y_train, y_pred_train_xgb,squared=False)
print(f"Train R2: {xgbtrainscore}, Test R2: {xgbtestscore}")
print(f"Train RMSE: {xgbtrainscorermse}, Test RMSE: {xgbtestscorermse}")

```

    Validation score:  0.695766277728868
    Train R2: 0.8757591338741197, Test R2: 0.7356445785398873
    Train RMSE: 0.15527768270536146, Test RMSE: 0.23252674311672142



```python
# Defining a hyperparameter grid for XGB
xgb_grid = {
    "xgb__n_estimators": [1000,1500,3000],
    "xgb__max_depth": [6,9,12,15],
    "xgb__learning_rate":[0.5, 0.1,.001],
    "xgb__gamma":[0.01, .1, .001]
}
```


```python
# Defining a GridSearchCV object to search for the best hyperparameters for the pipeline with the hyperparamater grid
xgb_pipe = GridSearchCV(estimator=pipe_xgb, param_grid=xgb_grid,n_jobs=-1, cv=5)

# Fitting the pipeline on the training data
xgb_model = xgb_pipe.fit(X_train,y_train)

# Printing validation score and best hyperparemeters
print("Validation score: ", xgb_pipe.best_score_)
print("Best hyperparameters: ", xgb_pipe.best_params_)
#Test data accuracy score 
y_pred_test_xgb = xgb_pipe.best_estimator_.predict(X_test)
xgbtestscore = r2_score(y_test, y_pred_test_xgb)
xgbtestscorermse = mean_squared_error(y_test, y_pred_test_xgb,squared=False)
#Train data accuracy score
y_pred_train_xgb = xgb_pipe.best_estimator_.predict(X_train)
xgbtrainscore = r2_score(y_train, y_pred_train_xgb)
xgbtrainscorermse = mean_squared_error(y_train, y_pred_train_xgb, squared=False)
print(f"Train R2: {xgbtrainscore}, Test R2: {xgbtestscore}")
print(f"Train RMSE: {xgbtrainscorermse}, Test RMSE: {xgbtestscorermse}")

```

    Validation score:  0.7241888963368177
    Best hyperparameters:  {'xgb__gamma': 0.001, 'xgb__learning_rate': 0.1, 'xgb__max_depth': 6, 'xgb__n_estimators': 1500}
    Train R2: 0.9518495587658475, Test R2: 0.7545110088939004
    Train RMSE: 0.09666671084087529, Test RMSE: 0.22407572410210705



```python
pipe_xgb2 = Pipeline([('preprocessing',preprocessor),('xgb', xgb.XGBRegressor(gamma=0.001,learning_rate=0.1,max_depth=6, n_estimators=1500))])

pipe_xgb2.fit(X_train, y_train)
```




    Pipeline(steps=[('preprocessing',
                     ColumnTransformer(transformers=[('log',
                                                      FunctionTransformer(func=<function <lambda> at 0x7fc92c0f1f70>),
                                                      ['Astringency', 'Body',
                                                       'Alcohol', 'Bitter', 'Sour',
                                                       'Fruits', 'Hoppy',
                                                       'Spices']),
                                                     ('scaler', StandardScaler(),
                                                      ['ABV', 'Min IBU', 'Max IBU',
                                                       'Astringency', 'Body',
                                                       'Alcohol', 'Bitter', 'Sweet',
                                                       'Sour', 'Fruits', 'Hoppy',
                                                       'Spices', 'Malty']...
                                  feature_types=None, gamma=0.001, gpu_id=None,
                                  grow_policy=None, importance_type=None,
                                  interaction_constraints=None, learning_rate=0.1,
                                  max_bin=None, max_cat_threshold=None,
                                  max_cat_to_onehot=None, max_delta_step=None,
                                  max_depth=6, max_leaves=None,
                                  min_child_weight=None, missing=nan,
                                  monotone_constraints=None, n_estimators=1500,
                                  n_jobs=None, num_parallel_tree=None,
                                  predictor=None, random_state=None, ...))])



I received good RMSE score for this model, I will go ahead and investigate feature importances.


```python
importance = pipe_xgb2.named_steps['xgb'].feature_importances_
feature_names = pipe_xgb2.named_steps['preprocessing'].transformers_[0][2] + pipe_xgb2.named_steps['preprocessing'].transformers_[1][2] + pipe_xgb2.named_steps['preprocessing'].transformers_[2][2]
# Add the feature names to the importance scores
importance_with_names = zip(feature_names, importance)
# Sort the features by importance
sorted_importance = sorted(importance_with_names, key=lambda x: x[1], reverse=True)
# Plot feature importance with names
pyplot.bar([x[0] for x in sorted_importance], [x[1] for x in sorted_importance])
pyplot.xticks(rotation=90)
pyplot.show()
```


    
![png](output_57_0.png)
    


Top 5 important features while predicting Rating as below.


```python
sorted_importance
```




    [('ABV', 0.01160989),
     ('Sour', 0.003091867),
     ('Fruits', 0.0029616235),
     ('Max IBU', 0.0028855458),
     ('Malty', 0.0028503262)]



### 5th Model: Model Stacking


```python
# Defining estimators as models with best hyperparameters for stacking
estimators = [
    ('lr', LinearRegression()),
    ('xgb', xgb.XGBRegressor(gamma= 0.001,learning_rate= 0.1, n_estimators= 1500, max_depth=6))
]

# Defining Stacking Regressor
sr = StackingRegressor(estimators)
```


```python
# Creating a pipeline with preprocessing step and model stacking
stacked = Pipeline([('preprocessing',preprocessor), ('model', sr)])

# Fitting the train data to pipeline
stacked.fit(X_train, y_train)

```




    Pipeline(steps=[('preprocessing',
                     ColumnTransformer(transformers=[('log',
                                                      FunctionTransformer(func=<function <lambda> at 0x7fc94c9f4c10>),
                                                      ['Astringency', 'Body',
                                                       'Alcohol', 'Bitter', 'Sour',
                                                       'Fruits', 'Hoppy',
                                                       'Spices']),
                                                     ('scaler', StandardScaler(),
                                                      ['ABV', 'Min IBU', 'Max IBU',
                                                       'Astringency', 'Body',
                                                       'Alcohol', 'Bitter', 'Sweet',
                                                       'Sour', 'Fruits', 'Hoppy',
                                                       'Spices', 'Malty']...
                                                                 gpu_id=None,
                                                                 grow_policy=None,
                                                                 importance_type=None,
                                                                 interaction_constraints=None,
                                                                 learning_rate=0.1,
                                                                 max_bin=None,
                                                                 max_cat_threshold=None,
                                                                 max_cat_to_onehot=None,
                                                                 max_delta_step=None,
                                                                 max_depth=6,
                                                                 max_leaves=None,
                                                                 min_child_weight=None,
                                                                 missing=nan,
                                                                 monotone_constraints=None,
                                                                 n_estimators=1500,
                                                                 n_jobs=None,
                                                                 num_parallel_tree=None,
                                                                 predictor=None,
                                                                 random_state=None, ...))]))])




```python
#Model stacking test data accuracy score
y_pred_test_stacked = stacked.predict(X_test)
stackedtestscore = r2_score(y_test, y_pred_test_stacked)
stackedtestscorermse = mean_squared_error(y_test, y_pred_test_stacked, squared=False)
#Train data accuracy score
y_pred_train_stacked = stacked.predict(X_train)
stackedtrainscore = r2_score(y_train, y_pred_train_stacked)
stackedtrainscorermse = mean_squared_error(y_train, y_pred_train_stacked, squared=False)

print(f"Train R2: {stackedtrainscore}, Test R2: {stackedtestscore}")
print(f"Train RMSE: {stackedtrainscorermse}, Test RMSE: {stackedtestscorermse}")
```

    Train R2: 0.9433176307305954, Test R2: 0.7765407066891912
    Train RMSE: 0.10488196213799642, Test RMSE: 0.2137853831510045


### Model Comparison
From all the models so far, stacked model for LR and XGB is the best model for average beer rating prediction since it has the least RMSE score.

## Beer Recommendation Systems

In this section, I will use cosine similarity and euclidean distance to recommend similar beers.

### Cosine Similarity


```python
# Transforming data using the preprocessor
X = preprocessor.fit_transform(df_pp)

# Computing pairwise cosine similarity matrix
cosine_sim_matrix = cosine_similarity(X)

# Defining function to get top 5 similar beers
def get_top_similar_beers(beer_name):
    idx = df[df['Name'] == beer_name].index[0]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    beer_indices = [i[0] for i in sim_scores]
    return df.drop(columns=['key','Style Key', 'Description']).iloc[beer_indices]

```


```python
# Below codes can be used to search beers and breweries within dataset.
# df[df['Brewery'].str.contains('')]
# df[df['Name'].str.contains('')]
```

With the created function, I can recommend 5 similar beers to a given beer using cosine similarity.


```python
get_top_similar_beers('Elephant Beer')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Style</th>
      <th>Brewery</th>
      <th>ABV</th>
      <th>Ave Rating</th>
      <th>Min IBU</th>
      <th>Max IBU</th>
      <th>Astringency</th>
      <th>Body</th>
      <th>Alcohol</th>
      <th>Bitter</th>
      <th>Sweet</th>
      <th>Sour</th>
      <th>Salty</th>
      <th>Fruits</th>
      <th>Hoppy</th>
      <th>Spices</th>
      <th>Malty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2206</th>
      <td>Hevelius Kaper</td>
      <td>Lager - European Strong</td>
      <td>Elbrewery Co. Ltd. Sp. z o.o.</td>
      <td>8.7</td>
      <td>3.42</td>
      <td>15</td>
      <td>40</td>
      <td>10</td>
      <td>27</td>
      <td>49</td>
      <td>29</td>
      <td>68</td>
      <td>31</td>
      <td>2</td>
      <td>28</td>
      <td>36</td>
      <td>9</td>
      <td>73</td>
    </tr>
    <tr>
      <th>2230</th>
      <td>Debowe MocneTyskie Browary Książęce (SABMiller)</td>
      <td>Lager - European Strong</td>
      <td>Tyskie Browary Książęce (SABMiller)</td>
      <td>7.0</td>
      <td>3.22</td>
      <td>15</td>
      <td>40</td>
      <td>10</td>
      <td>29</td>
      <td>29</td>
      <td>35</td>
      <td>36</td>
      <td>14</td>
      <td>1</td>
      <td>12</td>
      <td>42</td>
      <td>3</td>
      <td>50</td>
    </tr>
    <tr>
      <th>2231</th>
      <td>Lomza Mocne</td>
      <td>Lager - European Strong</td>
      <td>Browar Łomża Sp. z o.o.</td>
      <td>7.8</td>
      <td>3.10</td>
      <td>15</td>
      <td>40</td>
      <td>19</td>
      <td>33</td>
      <td>42</td>
      <td>33</td>
      <td>82</td>
      <td>31</td>
      <td>1</td>
      <td>44</td>
      <td>55</td>
      <td>8</td>
      <td>82</td>
    </tr>
    <tr>
      <th>2207</th>
      <td>Bavaria 8.6 Original</td>
      <td>Lager - European Strong</td>
      <td>Swinkels Family Brewers</td>
      <td>7.9</td>
      <td>2.56</td>
      <td>15</td>
      <td>40</td>
      <td>8</td>
      <td>35</td>
      <td>45</td>
      <td>19</td>
      <td>65</td>
      <td>20</td>
      <td>0</td>
      <td>22</td>
      <td>30</td>
      <td>8</td>
      <td>52</td>
    </tr>
    <tr>
      <th>2243</th>
      <td>Slavutich Mitzne</td>
      <td>Lager - European Strong</td>
      <td>Slavutych Brewery</td>
      <td>7.2</td>
      <td>2.78</td>
      <td>15</td>
      <td>40</td>
      <td>17</td>
      <td>35</td>
      <td>45</td>
      <td>24</td>
      <td>56</td>
      <td>27</td>
      <td>3</td>
      <td>30</td>
      <td>28</td>
      <td>5</td>
      <td>64</td>
    </tr>
  </tbody>
</table>
</div>



### Euclidean Distance


```python
# Applying the preprocessor to the data
X = preprocessor.fit_transform(df_pp)

# Creating a Nearest Neighbors model using the Euclidean distance metric
model = NearestNeighbors(metric='euclidean')

# Fit the model on the preprocessed data
model.fit(X)

def recommend_beers(beer_name):
    # Find the index of the input beer
    index = df[df['Name'] == beer_name].index[0]
    
    # Get the preprocessed data for the input beer
    beer_data = X[index].reshape(1, -1)
    
    # Find the 5 nearest beers based on Euclidean distance
    distances, indices = model.kneighbors(beer_data, n_neighbors=6)
    
    # Return the top 5 recommended beers (excluding the input beer itself)
    recommended_beers = df.iloc[indices.flatten()][1:]
    
    return recommended_beers
```

With the created function, I can recommend 5 similar beers to a given beer using cosine similarity.


```python
recommend_beers('Elephant Beer')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>key</th>
      <th>Style</th>
      <th>Style Key</th>
      <th>Brewery</th>
      <th>Description</th>
      <th>ABV</th>
      <th>Ave Rating</th>
      <th>Min IBU</th>
      <th>Max IBU</th>
      <th>...</th>
      <th>Body</th>
      <th>Alcohol</th>
      <th>Bitter</th>
      <th>Sweet</th>
      <th>Sour</th>
      <th>Salty</th>
      <th>Fruits</th>
      <th>Hoppy</th>
      <th>Spices</th>
      <th>Malty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2230</th>
      <td>Debowe MocneTyskie Browary Książęce (SABMiller)</td>
      <td>2481</td>
      <td>Lager - European Strong</td>
      <td>58</td>
      <td>Tyskie Browary Książęce (SABMiller)</td>
      <td>Notes:</td>
      <td>7.0</td>
      <td>3.22</td>
      <td>15</td>
      <td>40</td>
      <td>...</td>
      <td>29</td>
      <td>29</td>
      <td>35</td>
      <td>36</td>
      <td>14</td>
      <td>1</td>
      <td>12</td>
      <td>42</td>
      <td>3</td>
      <td>50</td>
    </tr>
    <tr>
      <th>2206</th>
      <td>Hevelius Kaper</td>
      <td>2457</td>
      <td>Lager - European Strong</td>
      <td>58</td>
      <td>Elbrewery Co. Ltd. Sp. z o.o.</td>
      <td>Notes:</td>
      <td>8.7</td>
      <td>3.42</td>
      <td>15</td>
      <td>40</td>
      <td>...</td>
      <td>27</td>
      <td>49</td>
      <td>29</td>
      <td>68</td>
      <td>31</td>
      <td>2</td>
      <td>28</td>
      <td>36</td>
      <td>9</td>
      <td>73</td>
    </tr>
    <tr>
      <th>2207</th>
      <td>Bavaria 8.6 Original</td>
      <td>2458</td>
      <td>Lager - European Strong</td>
      <td>58</td>
      <td>Swinkels Family Brewers</td>
      <td>Notes:</td>
      <td>7.9</td>
      <td>2.56</td>
      <td>15</td>
      <td>40</td>
      <td>...</td>
      <td>35</td>
      <td>45</td>
      <td>19</td>
      <td>65</td>
      <td>20</td>
      <td>0</td>
      <td>22</td>
      <td>30</td>
      <td>8</td>
      <td>52</td>
    </tr>
    <tr>
      <th>2201</th>
      <td>Baltika #9 Extra (Strong)</td>
      <td>2452</td>
      <td>Lager - European Strong</td>
      <td>58</td>
      <td>Baltika Breweries</td>
      <td>Notes:</td>
      <td>8.0</td>
      <td>2.85</td>
      <td>15</td>
      <td>40</td>
      <td>...</td>
      <td>23</td>
      <td>50</td>
      <td>21</td>
      <td>76</td>
      <td>21</td>
      <td>1</td>
      <td>24</td>
      <td>32</td>
      <td>6</td>
      <td>66</td>
    </tr>
    <tr>
      <th>2243</th>
      <td>Slavutich Mitzne</td>
      <td>2494</td>
      <td>Lager - European Strong</td>
      <td>58</td>
      <td>Slavutych Brewery</td>
      <td>Notes:</td>
      <td>7.2</td>
      <td>2.78</td>
      <td>15</td>
      <td>40</td>
      <td>...</td>
      <td>35</td>
      <td>45</td>
      <td>24</td>
      <td>56</td>
      <td>27</td>
      <td>3</td>
      <td>30</td>
      <td>28</td>
      <td>5</td>
      <td>64</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



## Project Summary

This project aimed to develop a beer rating predictor and recommendation system using Beer dataset. The project included an exploratory data analysis phase, followed by the creation of baseline models and more advanced models using pipelines and grid searches. The performance of these models was evaluated on a hold-out test set, and the best-performing model was identified for genre classification.
\
\
After investigating the data, regression models were created. Five different pipelines were created with different classifiers, and hyperparameter tuning was used to improve the accuracy scores. Stacked model for XG Boost and Linear Regression was identified as the best model for beer rating prediction with a RMSE score of 0.2137.
\
\
The project also identified the best predictors for beer rating as ABV, fruitiness and bitterness. 
\
\
Finally, 2 recommendation systems were created to recommend 5 similar beers for a given beer using cosine similarity and euclidean distance. 
\
\
Overall, the project successfully developed a beer rating predictor and recommendation system that can be used by Rum & Lemonade Bistro to brew their new beer in the best way possible and provided insights into which features should be targeted to maximize profits.
