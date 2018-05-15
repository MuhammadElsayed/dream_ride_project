
# coding: utf-8


#Loading libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0)
import seaborn as sns
from scipy import stats
from scipy.stats import norm


#loading data
all_data = pd.read_csv("data.csv")
train = all_data.iloc[:500000, :]
test = all_data.iloc[500000:, :]



train.head()



print ('The train data has {0} rows and {1} columns'.format(train.shape[0],train.shape[1]))
print ('----------------------------')
print ('The test data has {0} rows and {1} columns'.format(test.shape[0],test.shape[1]))


train.info()



#check missing values
train.columns[train.isnull().any()]



#missing value counts in each of these columns
miss = train.isnull().sum()/len(train)
miss = miss[miss > 0]
miss.sort_values(inplace=True)
miss


#SalePrice
sns.distplot(train['Price'])


#skewness
print("The skewness of SalePrice is {}".format(train['Price'].skew()))



#now transforming the target variable
target = np.log(train['Price'])
print ('Skewness is', target.skew())
sns.distplot(target)


#separate variables into new data frames
numeric_data = train.select_dtypes(include=[np.number])
cat_data = train.select_dtypes(exclude=[np.number])
print("There are {} numeric and {} categorical columns in train data".format(numeric_data.shape[1],cat_data.shape[1]))


#correlation plot
corr = numeric_data.corr()
sns.heatmap(corr)


print (corr['Price'].sort_values(ascending=False), '\n')



train['Year'].unique()



#let's check the mean price per quality and plot it.
pivot = train.pivot_table(index='Year', values='Price', aggfunc=np.median).sort_values(by='Price')
pivot


pivot.plot(kind='bar', color='red')



#GrLivArea variable
sns.jointplot(x=np.log(train['Mileage']), y=np.log(train['Price']))



cat_data.describe()



sp_pivot = train.pivot_table(index='Make', values='Price', aggfunc=np.median).sort_values(by='Price')
sp_pivot



sp_pivot.plot(kind='bar',color='red')



#create numeric plots
num = [f for f in train.columns if train.dtypes[f] != 'object']
nd = pd.melt(train, value_vars = num)
n1 = sns.FacetGrid (nd, col='variable', col_wrap=4, sharex=False, sharey = False)
n1 = n1.map(sns.distplot, 'value')
n1



def boxplot(x,y,**kwargs):
            sns.boxplot(x=x,y=y)
            x = plt.xticks(rotation=90)

cat = [f for f in train.columns if train.dtypes[f] == 'object']

p = pd.melt(train, id_vars='Price', value_vars=cat)
g = sns.FacetGrid (p, col='variable', col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(boxplot, 'value','Price')
g



#create a label set
label_df = pd.DataFrame(index = train.index, columns = ['Price'])
label_df['Price'] = np.log(train['Price'])
print("Training set size:", train.shape)
print("Test set size:", test.shape)


train_new = train
test_new = test
print ('Train', train_new.shape)
print ('----------------')
print ('Test', test_new.shape)



#get numeric features
numeric_features = [f for f in train_new.columns if train_new[f].dtype != object]

#transform the numeric features using log(x + 1)
from scipy.stats import skew
skewed = train_new[numeric_features].apply(lambda x: skew(x.dropna().astype(float)))
skewed = skewed[skewed > 0.75]
skewed = skewed.index
train_new[skewed] = np.log1p(train_new[skewed])
test_new[skewed] = np.log1p(test_new[skewed])
del test_new['Price']

