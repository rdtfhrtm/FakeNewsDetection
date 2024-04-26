#!/usr/bin/env python
# coding: utf-8

# # Goal: Build a system to predict fake news

# In[85]:


from IPython.display import Image
#Image(filename="C:/Users/ranja/Desktop/project/diagram.png")


# ###### Dataset used - https://www.kaggle.com/
# 
# ##### Dataset Description
# 
# train.csv: A full training dataset with the following attributes:
# 
# * id: unique id for a news article
# * title: the title of a news article
# * author: author of the news article
# * text: the text of the article
# * label: a label that marks the article as potentially fake/real
#   * 1: fake
#   * 0: real
# 
# test.csv: A testing training dataset with all the same attributes at train.csv without the label.
# 
# submit.csv: A sample file to see prediction results

# #### Importing all the required libraries

# In[86]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns 

import re #Regular expressions 
import nltk #Natural Language Toolkit
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
#from sklearn.linear_model import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
from prettytable import PrettyTable
import itertools
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


# In[87]:


nltk.download('stopwords')


# In[88]:


# Printing the stopwords in English
# We will have to remove these from the corpus during our analysis and pre-processing
print(stopwords.words('english'))


# 

# ## Data Pre-processing and Analysis

# In[89]:


from IPython.display import Image
#Image(filename="C:/Users/ranja/Desktop/project/text representation and tokenization.png")


# ##### Loading and exploratory data analysis : visualize the proportion of real and fake news.

# In[90]:


data = pd.read_csv(r'E:\Downloads\Kaggle Datasets\fake-news\train.csv')
conversion_dict = {0: 'Real', 1: 'Fake'}
data['label'] = data['label'].replace(conversion_dict)
data.label.value_counts()


# In[91]:


import plotly.express as px


# In[92]:


data.head()


# In[93]:


label = ['Real', 'Fake']
y1 = [10413, 10387]


# In[94]:


# function to add value labels
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i])


# In[95]:


get_ipython().run_line_magic('matplotlib', 'inline')
carrier_count = data["label"].value_counts()
sns.set(style="darkgrid")
sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
plt.title('Frequency Distribution of label')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Carrier', fontsize=12)
addlabels(label, y1) # calling the function to add lables
plt.show()


# In[96]:


type(carrier_count)


# In[138]:


plt.title('Proportion of Real vs. Fake News')
data["label"].value_counts().head(3).plot(kind = 'pie', autopct='%1.1f%%', figsize=(8, 8)).legend()


# ##### Visualizing top 10 Authors :-

# In[99]:


import seaborn as sns 
import matplotlib.pyplot as plt


# In[100]:


d = data['author'].value_counts().sort_values(ascending=False).head(10)
d = pd.DataFrame(d)
d = d.reset_index() # dataframe with top 5 authors

# Plotting
sns.set()
plt.figure(figsize=(20,6))
sns.barplot(x='index', y='author', data=d)
plt.xlabel("\n Authors")
plt.ylabel("Number of Articles written")
plt.title("Top 10 authors\n")
plt.show()


# ###### Checking for missing values

# In[101]:


data.isnull().sum() # missing values


# ###### replacing the missing values with empty strings

# In[102]:


#filling null values with empty string
data = data.fillna('') 
data.isnull().sum()


# #### we are using field text description to train our model to help predict if it is real or fake news.

# In[103]:


data.head()


# In[104]:


# Now we will separate the data and label i.e. text corpus and label fields
X = data.drop(columns='label', axis=1)
Y = data['label']


# In[105]:


X = data['text']
print(X)


# In[106]:


print(Y)


# In[107]:


X.head()


# ### removing special characters, stopwords, applying Stemming & Tf-IDF

# In[108]:


ps = PorterStemmer()

def stemming(corpus):
    # Pick all alphabet characters - lowercase and uppercase...all others such as numbers and punctuations will be removed. Numbers or punctuations will be replaced by a whitespace
    stemmed_corpus = re.sub('[^a-zA-Z]',' ',corpus)
    
    # Converting all letters to lowercase 
    stemmed_corpus = stemmed_corpus.lower()
    
    # Converting all to a splitted case or a list
    stemmed_corpus = stemmed_corpus.split()
    
    # Applying stemming, so we get the root words wherever possible + remove stopwords as well
    stemmed_corpus = [ps.stem(word) for word in stemmed_corpus] #if not word in stopwords.words('english')]
    
    # Join all the words in final content
    stemmed_corpus = ' '.join(stemmed_corpus)
    return stemmed_corpus


# In[109]:


data['text'] = data['text'].apply(stemming)


# In[110]:


print(data['text'])


# In[111]:


# Separating data and label
X = data['text'].values
Y = data['label'].values


# In[112]:


type(X)


# In[113]:


Y


# In[114]:


#TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
vectorizer.fit(X)
X = vectorizer.transform(X)


# In[115]:


print(X)


# In[116]:


type(X)


# ### Modeling & Model Evaluation

# In[117]:


#Splitting the data into test and train datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=124)


# #### Training  Model1 : Passive Aggressive Classiifier

# In[118]:


from sklearn.linear_model import PassiveAggressiveClassifier

model1 = PassiveAggressiveClassifier(max_iter=50)
model1.fit(X_train, Y_train)

# from sklearn.naive_bayes import MultinomialNB
# classifier=MultinomialNB()
# classifier.fit(X, Y)
#print("Accuracy score on the test data:  %0.3f" %test_data_accuracy)

X_train_prediction1 = model1.predict(X_train)
training_data_accuracy1 = accuracy_score(X_train_prediction1, Y_train)

print("Accuracy score on the training data:  %0.3f" %training_data_accuracy1)

# Accuracy Score on Test Data
X_test_prediction1 = model1.predict(X_test)
test_data_accuracy1 = accuracy_score(X_test_prediction1, Y_test)

print("Accuracy score on the test data:  %0.3f" %test_data_accuracy1)


# ###### Function to plot confusion Matrix :-

# In[119]:



def plot_confusion_matrix(cm, 
                          classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[120]:


cm = metrics.confusion_matrix(Y_test, X_test_prediction1)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])
plt.grid(None)


# In[121]:


# Classification report for test data
print(classification_report(Y_test, X_test_prediction1))


# #### Training  Model2 : Multinomial Naive Bayes

# In[122]:


from sklearn.naive_bayes import MultinomialNB

model2 = MultinomialNB()
model2.fit(X_train, Y_train)
X_train_prediction2 = model2.predict(X_train)
training_data_accuracy2 = accuracy_score(X_train_prediction2, Y_train)
print("Accuracy score on the training data:  %0.3f" %training_data_accuracy2)

 # Accuracy Score on Test Data
X_test_prediction2 = model2.predict(X_test)
test_data_accuracy2 = accuracy_score(X_test_prediction2, Y_test)

print("Accuracy score on the test data:  %0.3f" %test_data_accuracy2)


cm2 = metrics.confusion_matrix(Y_test, X_test_prediction2)
plot_confusion_matrix(cm2, classes=['Fake', 'Real'])
plt.grid(None)

print(classification_report(Y_test, X_test_prediction2))

 #### Training  Model3 : Logistic Regression

# In[123]:


#Logistic Regression model
from sklearn.linear_model import LogisticRegression
model3 = LogisticRegression()
model3.fit(X_train, Y_train)

X_train_prediction3 = model3.predict(X_train)
training_data_accuracy3 = accuracy_score(X_train_prediction3, Y_train)

print("Accuracy score on the training data:  %0.3f" %training_data_accuracy3)

# Accuracy Score on Test Data
X_test_prediction3 = model3.predict(X_test)
test_data_accuracy3 = accuracy_score(X_test_prediction3, Y_test)

print("Accuracy score on the test data:  %0.3f" %test_data_accuracy3)


# In[124]:


cm3 = metrics.confusion_matrix(Y_test, X_test_prediction3)
plot_confusion_matrix(cm3, classes=['Fake', 'Real'])
plt.grid(None)
#plt.axis('off')
#plt.rcParams["axes.grid"] = False


# In[125]:


# Classification report for test data
print(classification_report(Y_test, X_test_prediction3))


# #### Training Model4 : Random Forest

# In[126]:


from sklearn.ensemble import RandomForestClassifier
model4 = RandomForestClassifier ()
model4.fit(X_train, Y_train)

X_train_prediction4 = model4.predict(X_train)
training_data_accuracy4 = accuracy_score(X_train_prediction4, Y_train)

print("Accuracy score on the training data:  %0.3f" %training_data_accuracy4)

# Accuracy Score on Test Data
X_test_prediction4 = model4.predict(X_test)
test_data_accuracy4 = accuracy_score(X_test_prediction4, Y_test)

print("Accuracy score on the test data:  %0.3f" %test_data_accuracy4)


# In[129]:


cm4 = metrics.confusion_matrix(Y_test, X_test_prediction4)
plot_confusion_matrix(cm4, classes=['Fake', 'Real'])
plt.grid(None)
#plt.axis('off')
#plt.rcParams["axes.grid"] = False


# In[130]:


# Classification report for test data
print(classification_report(Y_test, X_test_prediction4))


# #### Training Model5 : Support Vector Machine

# In[131]:


from sklearn import svm
model5 = svm.SVC()
model5.fit(X_train, Y_train)

X_train_prediction5 = model5.predict(X_train)
training_data_accuracy5 = accuracy_score(X_train_prediction5, Y_train)

print("Accuracy score on the training data:  %0.3f" %training_data_accuracy5)

# Accuracy Score on Test Data
X_test_prediction5 = model5.predict(X_test)
test_data_accuracy5 = accuracy_score(X_test_prediction5, Y_test)

print("Accuracy score on the test data:  %0.3f" %test_data_accuracy5)


# In[132]:


cm5 = metrics.confusion_matrix(Y_test, X_test_prediction5)
plot_confusion_matrix(cm5, classes=['Fake', 'Real'])
plt.grid(None)

print(classification_report(Y_test, X_test_prediction5))
#plt.axis('off')
#plt.rcParams["axes.grid"] = False


# In[ ]:





# In[ ]:





# ##### Comparison of all algorithms Results

# In[133]:


from prettytable import PrettyTable
x = PrettyTable()
print('\n')
print("Comparison of all algorithm results")
x.field_names = ["Model", "Accuracy"]


x.add_row(["Passive Aggressive Classiifier", round(test_data_accuracy1,2)])
x.add_row(["Naive Bayes Algorithm", round(test_data_accuracy2,2)])
x.add_row(["LogisticRegression Algorithm", round(test_data_accuracy3,2)])
x.add_row(["Random Forest Algorithm", round(test_data_accuracy4,2)])
x.add_row(["Support Vector Machine", round(test_data_accuracy5,2)])

print(x)
print('\n')


# In[ ]:





# # Making a Prediction

# In[137]:


X_new = X_test[1000:1005]

prediction = model1.predict(X_new)
print(prediction)

print(len(prediction))

for i in range(len(prediction)):
    print("this news is ", str(prediction[i]))


if (prediction[0] == 'Real'):
  print('The is Real News')
else:
  print('The news is Fake')


# In[136]:


data[1000:1005]


# In[ ]:





