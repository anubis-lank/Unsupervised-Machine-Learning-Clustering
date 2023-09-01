# **DATASET PRE-PROCESSING**

import pandas as pd 

# **DATA VISUALIZATION**

import seaborn as sns 
import matplotlib.pyplot as plt

# **DATA PREPROCESSING**

from sklearn.preprocessing import MinMaxScaler

# **MACHINE LEARNING MODELS**

from sklearn.cluster import KMeans

# **METRICS**

from sklearn.metrics import silhouette_score

# **INPUT**
segmentation_data = pd.read_csv("C:/Users/**INPUT FILE PATH HERE**/segmentation data.csv")

# take a look at the first 10 rows of the data
segmentation_data.head(10)

# describe shows statistical information like mean, standard deviation, min & max, etc.
segmentation_data.describe()

# show unique values to provide better understanding of the categorical type data
segmentation_data.nunique()

# this data is well structured and most of the categorical data has already been converted to numerical data
# which will be better for the machine learning model.

# DATA CLEANING

# check for null values
segmentation_data.info()

segmentation_data.isna().sum()
# no null values to be found

# the ID column is not need for the cluster and provides no impact into customer identification
# we will drop this column
segmentation_data.drop(["ID"], inplace = True, axis = 1)

segmentation_data.head()

# next let's take a closer look at Age and Income
plt.figure(figsize = (21, 15))

plt.subplot2grid((2,2), (0,0))
box1 = sns.boxplot(y = segmentation_data.Age)
plt.title("Age")

plt.subplot2grid((2,2), (0,1))
box2 = sns.boxplot(y = segmentation_data.Income)
plt.title("Income")

plt.show()
# these box plots show plenty of outliers in both of these features
# we will clean up the outliers and the scale the data

#delete_outlier(segmentation_data["Age"])
#delete_outlier(segmentation_data["Income"])

# display updated box plots
plt.figure(figsize = (21, 15))

plt.subplot2grid((2,2), (0,0))
box1 = sns.boxplot(y = segmentation_data.Age)
plt.title("Age")

plt.subplot2grid((2,2), (0,1))
box2 = sns.boxplot(y = segmentation_data.Income)
plt.title("Income")

plt.show()
# this looks much better, now to scale the data in both of these features

scaler = MinMaxScaler()
segmentation_data["Age"] = scaler.fit_transform(segmentation_data[["Age"]])
segmentation_data["Income"] = scaler.fit_transform(segmentation_data[["Income"]])

# lets visualise the changes made to the two features
segmentation_data.head(10)

sns.displot(segmentation_data, x= segmentation_data["Age"], kind= 'kde', fill= True)
sns.displot(segmentation_data, x= segmentation_data["Income"], kind= 'kde', fill= True)
# the data has been cleaned and scaled which will help with the accuracy and process time for the ML algorithm later

segmentation_data.describe()
# the min & max for Sex, Marital status, Age and Income are now 0 and 1
# but for Education, Occupation, Settlement Size the min & max varies from 0 to 2/3
# scaling these three features would reduce segmentation bias 

segmentation_data["Education"] = scaler.fit_transform(segmentation_data[["Education"]])
segmentation_data["Occupation"] = scaler.fit_transform(segmentation_data[["Occupation"]])
segmentation_data["Settlement size"] = scaler.fit_transform(segmentation_data[["Settlement size"]])

# lets take a final look at the dataset after all the changes
segmentation_data.head(10)
segmentation_data.describe()

# the dataset is now ready to be labelled
# lets create our model and fit the data

kmeans = KMeans()
kmeans.fit(segmentation_data)

# let's take a look at the silhoutte score for the model

score = silhouette_score(segmentation_data, kmeans.labels_)
print("Silhouette score: {:.3f}".format(score))

# lets see if we can change the parameters of the K means to achieve a better result

# below is a Within_Cluster_Sum_of_Squares table
# this shows how close the data is to each other within each cluster
# the sum of squares calculates how far from the center of the cluster each data point is
# the idea is we want our data to be as close to the center as possible

wcss_table = {'score':[], 'no_of_clusters':[]}
for i in range(1,11):
    kmeans = KMeans(i, random_state=0)
    kmeans.fit(segmentation_data)
    wcss_table['score'].append(kmeans.inertia_)
    wcss_table['no_of_clusters'].append(i)

wcss_table_df = pd.DataFrame(wcss_table)
wcss_table_df.head(10)

# lets plot the above points on a graph to locate the elbow point of WCSS

plt.figure(figsize=(14,10))
plt.plot(wcss_table_df['no_of_clusters'], wcss_table_df['score'], marker='o')
plt.title('Elbow method to determine number of clusters(K)')
plt.show()

# to me from the graph above using clusters(k) of 6 could improve our score

kmeans_6 = KMeans(n_clusters=6)
kmeans_6.fit(segmentation_data)

score_6 = silhouette_score(segmentation_data, kmeans_6.labels_)
print("Silhouette score: {:.3f}".format(score_6))

# there is a .01 increase for the silhouttes score
# lets visulaise the clusters

# create a new data set which is a duplicate of the original dataset with an added label colum

# create a list of the labels for each row of our data
predictions = kmeans_6.fit_predict(segmentation_data)

labelled_seg_data = segmentation_data.copy()
labelled_seg_data["label"] = predictions

labelled_seg_data.head(10)
# now we can see what the model labelled each row of the data

sns.scatterplot(x=labelled_seg_data["Age"],
                y=labelled_seg_data["Income"],
                hue=labelled_seg_data["label"])

# decided to go back an not take out the outliers in the Age and Income features
# ran the Kmeans model on this data and achieved a silhoutte score of 0.414 which is a big improvement
# using 6 cluster(k) the score hits .437