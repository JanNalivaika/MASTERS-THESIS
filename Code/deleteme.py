from sklearn.preprocessing import minmax_scale
import numpy as np
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

x_array = np.array([-35,-65,-95,-135])

a= min_max_scaler.fit_transform(x_array.reshape(-1, 1))*100

print(a)


