import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

inp_dir='/Users/shrishaa/Documents/computer vision/Car_parking/data'
categories=['empty','not_empty']

data=[]
labels=[]

for cat_indx,cat in enumerate(categories):
    for file in os.listdir(os.path.join(inp_dir,cat)):
        imp_path=os.path.join(inp_dir,cat,file)
        img=imread(imp_path)
        img=resize(img,(15,15))
        data.append(img.flatten())
        labels.append(cat_indx)


data=np.asarray(data)
labels=np.asarray(labels)


x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.2,shuffle=True,stratify=labels)

classifier=SVC()

parameters=[{'gamma':[0.01,0.001,0.0001],'C':[1,10,100,1000]}]

grid_search=GridSearchCV(classifier,parameters)

grid_search.fit(x_train,y_train)

best=grid_search.best_estimator_

y_pred=best.predict(x_test)

acc=accuracy_score(y_pred,y_test)

pickle.dump(best,open('./classifier.p','wb'))



