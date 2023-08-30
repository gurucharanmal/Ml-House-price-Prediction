
'''
#prerequisite 
alredy install librarys 
1, numpy
2. pandas
3. sklearn

#good network connection

 '''
 

#first import neeedful library's

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


def main():   # main() function to control all the projest
    
         #data collection
        obj=fetch_california_housing()
        df=pd.DataFrame(obj.data,columns=obj.feature_names)
        
        df['Target']=obj.target    #add target data in our DataFrame
        
        
        from geopy.geocoders import Nominatim      #import geopy library to collect 'raod' and 'county' data by usnig latitude and longitude
        geolocator = Nominatim(user_agent='geopyiExrcise')
        
        latitude=np.array([df.iloc[:,6].values])       #take latitude data from DataFrame
        longitude = np.array([df.iloc[:,7].values])    #take longitude data from DataFrame
        
        
        def location(latitude,longitude,list_road,list_county):      # the location funtion return the 'road' and 'county' values using geopy library
          latitude=str(latitude[0])
          longitude=str(longitude[0])
          try:
            loc_info = loc=geolocator.reverse(latitude + ','+ longitude).raw['address']
        
            if loc_info is not None:
              if loc_info['road']:
                list_road.append(loc_info['road'])
              else:
                list_road.append(None)
              if loc_info['county']:
                list_county.append(loc_info['county'])
              else:
                list_county.append(None)
            else:
                list_road.append(None)
                list_county.append(None)
                
          except:
            list_road.append(None)
            list_county.append(None)
        
        
        list_road=[]                 #this all line 49 to 62 takes lot of time someting 3 to 4 hour's , becouse of , to collect 'road and county data'
        list_county=[]
        for i in range(0,20641):                          #in that pint make sore your netwoek will good
             location(latitude[:,i],longitude[:,i],list_road,list_county)
        
        new_list_r=list_road
        new_list_c=list_county
        
        
        dct={'road':new_list_r,'county':new_list_c}
        new_df=pd.DataFrame(dct)
        
        dct={'road':new_list_r,'county':new_list_c}
        new_df=pd.DataFrame(dct)
        
        
        
        
        #adding new values 'road and county' in our DataFrame
        
        new_df=pd.read_csv('/content/road_and_county.csv')   
        df['road']=new_df.iloc[:,0].values
        df['county']=new_df.iloc[:,1].values
        df=df.sample(axis=0,frac=1)
        
        
        #from line 75 to 90 finding missing values in road column , becous in 'road column' there are many Nan values
        
        column_to_check = 'road'
        
        # Find the indices of NaN values in the specified column
        missing_road_values = df[df[column_to_check].isna()].index
        
        column_to_check = 'county'
        
        # Find the indices of NaN values in the specified column
        missing_county_values = df[df[column_to_check].isna()].index
        
        missing_road_xtrain=np.array([[df['MedInc'][i] ,df['AveRooms'][i] , df['AveBedrms'][i]]  for i in range(df.shape[0])  if i not in missing_road_values])
        missing_road_ytrian=np.array([  df['road'][i] for i in range(df.shape[0]) if i not in missing_road_values ])
        missing_road_xtest=np.array([[df['MedInc'][i] ,df['AveRooms'][i] , df['AveBedrms'][i]]  for i in range(df.shape[0])  if i in missing_road_values])
        
        
        
        # now using SGDClassifier algorithm to predict Nan values of 'road' using known values
        from sklearn.linear_model import SGDClassifier
        
        #model initialization
        model1=SGDClassifier()
        
        #training the model for road values 
        model1.fit(missing_road_xtrain,missing_road_ytrian)
        
        #y_prediction
        missing_road_y_pred=model1.predict(missing_road_xtest)
        
        
        #do same thing for 'county' values ,line 106 to 121, as we did for 'road' column
        missing_county_xtrain=np.array([[df['MedInc'][i] ,df['AveRooms'][i] , df['AveBedrms'][i]]  for i in range(df.shape[0])  if i not in missing_county_values])
        missing_county_ytrian=np.array([  df['road'][i] for i in range(df.shape[0]) if i not in missing_county_values ])
        missing_county_xtest=np.array([[df['MedInc'][i] ,df['AveRooms'][i] , df['AveBedrms'][i]]  for i in range(df.shape[0])  if i in missing_county_values])
        
        
        from sklearn.linear_model import SGDClassifier
        
        #model initialization
        model2=SGDClassifier()
        
        #training the model for county values 
        model2.fit(missing_county_xtrain,missing_county_ytrian)
        
        #y_prediction
        missing_county_y_pred=model1.predict(missing_county_xtest)
        
        
        #to add predicted values of road and county in DataFrame
        
        for n,i in enumerate(missing_road_values):
          df['road'][i]=missing_road_y_pred[n]
        
        for n,i in enumerate(missing_county_values):
          df['county'][i]=missing_county_y_pred[n]
        
        
        #import LabelEncoder to convert string values of 'road and county' values to intiger
        from sklearn.preprocessing import LabelEncoder
        
        le=LabelEncoder()
        df['road']=le.fit_transform(df['road'].values)
        df['county']=le.fit_transform(df['county'].values)
        
        #dorp latitude and longitude columns
        df=df.drop(labels=['Latitude'	,'Longitude'],axis=1)
        
        
        
        ##taking data to build our model 

        y_data=df['Target'].values           # target values for y_data
        df=df.drop(labels=['Target'],axis=1)
        
        x_data=df.iloc[:,:].values
        
        
        #import train_test_split method to split data in two part 
        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.1)
        
        
        #import RandomForestRegressor to train or build our model
        from sklearn.ensemble import RandomForestRegressor
        
        model=RandomForestRegressor()
        
        #trian our model
        model.fit(x_train,y_train)
        
        #predict y data using model
        y_pred=model.predict(x_test)
        
        from sklearn.metrics import r2_score
        
        #cheak accuracy of model using r2_score
        r2=r2_score(y_pred,y_test)
        print('our model is ', r2*100, "% accurate ",sep='')
        
        
        #now model redy to use.....
        
        op=model.predict(np.array([[4.1339 ,	34.0 ,	4.950413 ,	1.000000 , 	948.0 ,	2.611570 ,  5040	 , 34]]))
        print(op)   # as example 


main()   # calling main() funtion


