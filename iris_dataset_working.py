#!/usr/bin/env python
# coding: utf-8

# In[263]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()
df=pd.DataFrame(iris.data, columns=iris.feature_names)
df['species']=df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
X= df.iloc[:,:-1]
Y=df['species']
df.head(5)


# In[264]:


# to display stats
df.describe()


# In[265]:


# display basic info 
df.info()


# In[266]:


#preprocessing 


# In[267]:


df.isnull().sum()


# In[268]:


# exploratory analysis 


# In[269]:


import numpy as np

feature_means = df.groupby('species').mean()

feature_means.plot(kind='bar', figsize=(10, 6))
plt.title("Average Feature Values for Each Species")
plt.ylabel("Mean Value")
plt.xticks(rotation=0)
plt.show()


# In[270]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.boxplot(x='species',y='sepal length (cm)',data=df)
plt.title('sepal Length by Species')
plt.show()


# In[271]:


df.corr()


# In[272]:


corr=df.corr()
fig,ax=plt.subplots(figsize=(5,4))
sns.heatmap(corr,annot=True,ax=ax,cmap='coolwarm')


# In[273]:


# label encoder


# In[274]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[275]:


df['species']=le.fit_transform(df['species'])
df.tail()


# In[276]:


# model training 


# In[277]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
from sklearn.datasets import load_iris
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target  # Already encoded (0,1,2)

# Features & target
X = df.iloc[:, :-1]  # first 4 columns
y = df['species']    # already numeric


# In[278]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[279]:


model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))


# In[280]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[281]:


knn = KNeighborsClassifier(n_neighbors=3)  # Try k=3, 5, 7
knn.fit(X_train, y_train)


# In[282]:


y_pred = knn.predict(X_test)


# In[283]:


for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"k={k} -> Accuracy: {acc:.4f}")


# In[284]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Dictionary of models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    
    "Naive Bayes": GaussianNB(),
    
}

# Train and evaluate
print("Model Accuracies:\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name}: Accuracy = {acc:.4f}")


# In[285]:


from sklearn.model_selection import cross_val_score
import numpy as np

scores = cross_val_score(model, X, y, cv=10)
print("Cross-validated accuracy:", np.mean(scores))


# In[286]:


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X, y = iris.data, iris.target

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    
    
    "Naive Bayes": GaussianNB(),
    "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
    "Logistic Regression": LogisticRegression(max_iter=200)
}

print("Model-wise Cross-Validated Accuracies (5-fold):\n")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=10)
    print(f"{name}: mean={scores.mean():.4f}, std={scores.std():.4f}")


# In[287]:


import matplotlib.pyplot as plt

# Model names and their cross-validated mean accuracies
model_names = ['Decision Tree', 'Random Forest', 'Naive Bayes', 'KNN (k=7)', 'Logistic Regression']
accuracies = [0.9667, 0.9667, 0.9533, 0.9800, 0.9733]

# Create the bar plot
plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, accuracies, color='orange')
plt.ylim(0.9, 1.0)
plt.ylabel('Cross-Validated Accuracy')
plt.xlabel('Model')
plt.title('Model Comparison on Iris Dataset')

# Add accuracy values above bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height - 0.01, f"{height:.4f}", ha='center', color='black')

plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# # logistic regression 

# In[288]:


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score

iris=load_iris()

df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['species']=iris.target

X=df.iloc[:,:-1]
y=df['species']

X_train , X_test , y_train,y_test=train_test_split(X,y,test_size=0.2)
# Model
model=LogisticRegression(max_iter=200)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

# model ACC
accuracy=accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", accuracy)

# Test
#sample=[[5.9,3.0,5.1,1.8]] 

predicted_class=model.predict(sample)
print("predicted species:", iris.target_names[predicted_class])


# # knn

# In[296]:


import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris=load_iris()

df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['species']=iris.target
X=df.iloc[:,:-1]
y=df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model=KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print("KNN Accuracy:",accuracy)

#sample=[[5.9,3.0,5.1,1.8]] likely virginica
#sample=[[5.5, 2.3, 4.0, 1.3]] likely versicolor
#sample=[[5.1, 3.5, 1.4, 0.2]] likely setosa 

predicted_class = model.predict(sample)
print("Predicted species:", iris.target_names[predicted_class[0]])


# #random forest 

# In[290]:


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

iris =load_iris()

df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['species']=iris.target

X = df.iloc[:, :-1]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("Random Forest Accuracy:", accuracy)


#sample=[[5.9,3.0,5.1,1.8]] likely virginica
#sample=[[5.5, 2.3, 4.0, 1.3]] likely versicolor
#sample=[[5.1, 3.5, 1.4, 0.2]] likely setosa 

predicted_class = model.predict(sample)
print("Predicted species:", iris.target_names[predicted_class[0]])


# #decision tree 
# 

# In[291]:


import pandas as pd 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['species']=iris.target

X=df.iloc[:,:-1]
y=df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model=DecisionTreeClassifier()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print("Decision Tree Accuracy:", accuracy)

#sample=[[5.9,3.0,5.1,1.8]] likely virginica
#sample=[[5.5, 2.3, 4.0, 1.3]] likely versicolor
#sample=[[5.1, 3.5, 1.4, 0.2]]  likely setosa

predicted_class = model.predict(sample)
print("Predicted species:", iris.target_names[predicted_class[0]])


# #svm

# In[298]:


import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 

iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['species']=iris.target

X=df.iloc[:,:-1]
y=df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model=SVC()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print("SVM Accuracy:", accuracy)


#sample=[[5.9,3.0,5.1,1.8]] likely virginica
sample=[[5.5, 2.3, 4.0, 1.3]] 
#sample=[[5.1, 3.5, 1.4, 0.2]]  likely setosa

predicted_class = model.predict(sample)
print("Predicted species:", iris.target_names[predicted_class[0]])


# In[293]:


import matplotlib.pyplot as plt 
models = ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'SVM']
accuracies = [0.93, 0.96, 1.0, 0.96, 0.933]  

plt.figure(figsize=(10,6))
bars = plt.bar(models, accuracies, color='red', edgecolor='black')

for bar in bars:
    yval=bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0-0.15,yval + 0.005, f"{yval:.3f}", fontsize=10)
plt.title("Model Accuracy Comparison on Iris Dataset")
plt.xlabel("Machine Learning Algorithms")
plt.ylabel("Accuracy")
plt.ylim(0.9, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# In[ ]:





# In[ ]:





# In[ ]:




