#!/usr/bin/env python
# coding: utf-8

# In[1]:


#library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#menampilkan dan membaca data
data = pd.read_excel("diabetes_model.xlsx")


# In[3]:


data


# In[4]:


#mengganti kolom null seperti [glucose,bloodpressure,skinthickness,BMI,insulin] dengan nilai dengan rata-rata kolom masing masing
zero_not_accepted=['Glucose','BloodPressure','SkinThickness','BMI','Insulin']
for col in zero_not_accepted:
    data[col]=data[col].replace(0,np.NaN)
    mean=int(data[col].mean(skipna=True))
    data[col]=data[col].replace(np.NaN,mean)


# In[5]:


#mengambil variabel bebas
X = data.iloc[:,0:8]
X.head()


# In[6]:


#mengambil variabel terikat
Y = data.iloc[:,8]
Y.head()


# In[7]:


#membuat grafik
plt.figure(figsize=(25,7))
sns.countplot(x='Age',hue='Outcome',data=data)


# In[8]:


#memisahkan kumpulan data menjadi data training dan data testing 20%
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


# In[9]:


#melakukan normalisasi data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[10]:


#memanggil fungsi knn
classifier = KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean') 


# In[11]:


#fitting model (masukkan data training pada fungsi klasifikasi KNN)
classifier.fit(X_train,Y_train)


# In[12]:


#prediksi (menentukan hasil dari x_test dengan y prediksi)
Y_pred = classifier.predict(X_test)


# In[13]:


#menampilkan confusion matrix dari hasil prediksi dengan klasifikasi KNN
conf_matrix = confusion_matrix(Y_test,Y_pred)
print(conf_matrix)
print(f1_score(Y_test,Y_pred))


# In[14]:


#menampilkan klasifikasi report
print(classification_report(Y_test,Y_pred))


# In[15]:


#accuracy
print(accuracy_score(Y_test,Y_pred))


# In[16]:


#Nama: Nellis Neria Aurum Tectona
#NIM: A11.2020.12668
#Kelas: 4504


# In[ ]:




