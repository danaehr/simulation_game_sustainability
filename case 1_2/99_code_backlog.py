# -*- coding: utf-8 -*-
"""

@author: Daniel
"""
##################################################################################################
# seasonal forecasting with prophet

from prophet import Prophet


#prepare data for prediction model
data = pd.DataFrame(prices_cleaned[m][comp])
data['y']=data[comp]
data['ds']=data.index
del data[comp]

model = Prophet(yearly_seasonality=True)
model.fit(data)

future = model.make_future_dataframe(periods=6, freq='YE', include_history=True)

y_predict = model.predict(future)
plt.ioff()
model.plot(y_predict)
plt.title(label = f"{m} - {comp} - prediction - prophet")
plt.legend()                
plt.savefig(f"{output_dir}\{name_pattern}_prophet.png")
plt.show()