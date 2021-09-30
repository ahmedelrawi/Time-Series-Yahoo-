import numpy as np  
import pandas as pd
from numpy import array
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout


class Time_Price_Yahoo:

    def Data(self):
        self.data = pd.read_csv('yahoo_stock.csv')
        self.data.set_index('Date',inplace= True)
        self.data.index = pd.DatetimeIndex(self.data.index, freq='D')
        return self.data
    
    def train(self):
        self.Data()
        
        import math
        self.trained_data = self.data['High'].iloc[:-4]
        pd.DataFrame(self.trained_data)
        self.train_len = math.ceil(len(self.trained_data)*0.8)
        
        X_train=[]
        Y_train=[]
        for i in range(6, len(self.trained_data)):
            X_train.append(self.trained_data[i-6:i])
            Y_train.append(self.trained_data[i])

        self.X_train, self.Y_train= np.array(X_train), np.array(Y_train)
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))

        return self.train_len, self.X_train, self.Y_train

 
    def Learn(self):
        self.train()

        self.model=Sequential()
        self.model.add(LSTM(50, activation='relu', input_shape=(self.X_train.shape[1],1)))
        self.model.add(Dense(25))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.summary()
        self.model.fit(self.X_train, self.Y_train, epochs=10, batch_size=10, verbose=0)

    def Evaluate(self):
        self.Learn()
        
        self.test_data = self.trained_data[self.train_len-6:]
        X_val=[]
        Y_val=[] 

        for i in range(6, len(self.test_data)):
            X_val.append(self.test_data[i-6:i])
            Y_val.append(self.test_data[i])

        self.X_val, self.Y_val = np.array(X_val), np.array(Y_val)
        self.X_val = np.reshape(self.X_val, (self.X_val.shape[0], self.X_val.shape[1],1))

        return self.X_val, self.Y_val

    
    


    def Prediction(self):
        
        self.Evaluate()

        self.x_input = self.test_data[341:].values.reshape(1,-1)
        self.temp_input=list(self.x_input)
        self.temp_input=self.temp_input[0].tolist()
        self.day_new=np.arange(1,30)
        self.day_pred=np.arange(101,131)

        lst_output=[]
        n_steps=29
        i=0
        while(i<30):

            if(len(temp_input)>29):
              
                x_input=np.array(temp_input[1:])
                print("{} day input {}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = self.model.predict(x_input, verbose=0)
                print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                yhat = self.model.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i=i+1


        return lst_output








