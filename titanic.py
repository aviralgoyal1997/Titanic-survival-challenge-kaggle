import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('train.csv')
##to analyse gender vs dead
survived_sex = df[df['Survived']==1]['Sex'].value_counts()
dead_sex= df[df['Survived']==0]['Sex'].value_counts()
cx=pd.DataFrame([survived_sex,dead_sex])
cx.index=['survive','dead']
##cx.plot(kind='bar',stacked=True, figsize=(15,8))

plt.show()
##to analyse class vs dead
survived_class=df[df['Survived']==1]['Pclass'].value_counts()
dead_class=df[df['Survived']==0]['Pclass'].value_counts()
dx=pd.DataFrame([survived_class,dead_class])
dx.index=['survived','dead']
dx.plot(kind='bar',stacked=True,figsize=(15,8))
#plt.show()
##age vs ddead
df['Age']=df['Age'].fillna(df['Age'].median())
zx=df[df['Survived']==1]['Age']
zc=df[df['Survived']==0]['Age']
plt.hist([zx,zc],stacked=True,bins=30,label=['survivrd','dead'])
#plt.show()
##fare vs dead
zq=df[df['Survived']==1]['Fare']
zw=df[df['Survived']==0]['Fare']
plt.hist([zq,zw],stacked=True,bins=30,label=['survivrd','dead'])
#plt.show()
##fare and age vs dead
plt.figure(figsize=(15,8))
ax=plt.subplot()
ax.scatter(df[df['Survived']==1]['Age'],df[df['Survived']==1]['Fare'],c='Green',s=40)
ax.scatter(df[df['Survived']==0]['Age'],df[df['Survived']==0]['Fare'],c='red',s=40)
ax.set_xlabel('age')
ax.set_ylabel('fare')
ax.legend(('survived', 'dead'),scatterpoints=1,loc='upper right' ,fontsize=15)
#plt.show()
##ticketfare correlates with pclass
ax=plt.subplot()
ax.set_ylabel('Average fare')
xc=df.groupby('Pclass').mean()['Fare']
xc.plot(kind='bar',figsize=(15,8),ax=ax)
#plt.show()
## AS WE ANALYZED GENDER VS DEAD NOW ANALYZE EMBARKATION SITE VS DEAD
##now mix training and testinng data
def get_combined_data():
    train=pd.read_csv('train.csv')
    test=pd.read_csv('test.csv')
    target=train.Survived
    train.drop('Survived',1,inplace=True)
    combined=train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index',inplace=True,axis=1)
    return combined

combined=get_combined_data()
##FEATURE ENGINEERING

##to gain information from name we can use title like(sir,master,dona,etc) so extract title
s=combined['Name'].str.split(',',expand=True)
s.columns=['first name','title']
z=s['title'].str.split('.',expand=True)
z.columns=['title','asdf','wert']
combined['title']=z.title
##We have seen in the first part that the Age variable was missing 177 values. This is a large number ( ~ 13% of the dataset). Simply replacing them with the mean or the median age might not be the best solution since the age may differ by groups and categories of passengers.
##To understand why, let's group our dataset by sex, Title and passenger class and for each subset compute the median age.
grouped_train = combined.head(891).groupby(['Sex','Pclass','title'])
grouped_median_train = grouped_train.median()
grouped_test = combined.iloc[891:].groupby(['Sex','Pclass','title'])
grouped_median_test = grouped_test.median()

def process_name():
    global combined
    combined.drop('Name',axis=1,inplace=True)
    titles_dummies = pd.get_dummies(combined['title'],prefix='title')
    combined=pd.concat([combined,titles_dummies],axis=1)

    combined.drop('title',axis=1,inplace=True)





process_name()
#process fare
s=combined[combined['Fare'].isnull()].index
#s in test set so
p=combined.iloc[891:].Fare.mean()
combined.loc[s,'Fare']=p

#processing embarked
e=combined[combined['Embarked'].isnull()].index
for i in e:
    combined.loc[i,'Embarked']='S'
embarked_dummies=pd.get_dummies(combined['Embarked'],prefix='Embarked')
combined=pd.concat([combined,embarked_dummies],axis=1)
combined.drop('Embarked',axis=1,inplace=True)

##processing cabins
combined.Cabin.fillna('U', inplace=True)
combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])
cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')
combined = pd.concat([combined,cabin_dummies], axis=1)
combined.drop('Cabin', axis=1, inplace=True)

##age processing
combined.loc[0:891,'Age']=df['Age']
combined.loc[891:,'Age']=combined.iloc[891:]['Age'].median()

#processing sex

combined['Sex']=combined['Sex'].map({'male':1,'female':0})

#process pclass
pclass_dummies=pd.get_dummies(combined['Pclass'],prefix='Pclass')
combined=pd.concat([combined,pclass_dummies],axis=1)
combined.drop('Pclass',axis=1,inplace=True)


#process ticket

p=[]
for i in combined['Ticket']:
    i=i[0]
    p.append(i)
combined['Ticket']=p

q=[]
for i in p:
    if (i.isdigit()):
        q.append('XXX')
    else:
        q.append(i)

combined['Ticket']=q
ticket_dummies=pd.get_dummies(combined['Ticket'],prefix='Ticket')
combined=pd.concat([combined,ticket_dummies],axis=1)
combined.drop('Ticket',axis=1,inplace=True)
    
#process family
combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    
    
combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2<=s<=4 else 0)
combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5<=s else 0)

combined.drop('PassengerId',axis=1,inplace=True)

##MODELLING


train=combined.head(891)
test=combined.iloc[891:]
targets=df.Survived
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
clf=RandomForestClassifier(n_estimators=50,max_features='sqrt')
clf.fit(train,targets)
model=SelectFromModel(clf,prefit=True)
train_reduced=model.transform(train)
test_reduced=model.transform(test)
parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}

model=RandomForestClassifier(**parameters)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, targets, test_size=0.2)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

output=model.predict(test).astype(int)
final=pd.DataFrame()
xc=pd.read_csv('test.csv')
final['PassengerId']=xc['PassengerId']
final['Survived']=output
final[['PassengerId','Survived']].to_csv('titanicoutput.csv',index=False)
