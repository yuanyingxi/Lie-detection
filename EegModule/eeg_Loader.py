import os
from typing import Union

import numpy as np
import torch
import pandas as pd
from numpy.f2py.auxfuncs import throw_error
from sklearn.preprocessing import StandardScaler

from EegModule.processing import segment_data, extract_dwt_features
from .model import MyModel

class EEGFileProcessor:
    def __init__(self,model_path='EegModule/best_model.pth'):
        self.model = MyModel()
        self.model.load_state_dict(torch.load(model_path))



    def predict(self,input_path:pd.DataFrame,encoding='utf-8'):
        # 加载testData
        data_test = self.load_data(input_path,encoding)
        self.model.eval()
        with torch.no_grad():
            outputs =  self.model(data_test)
            predi = (outputs > 0.5).float()
        print("prediction:",predi)
        return predi

    #从文件加载data
    @classmethod
    def load_data(cls,df:pd.DataFrame,encoding='utf-8'):
        # df: pd.DataFrame
        # if os.path.isfile(path) and path.endswith('.csv'):
        #     df = pd.read_csv(os.path.join(os.getcwd(),path))
        #     df = df.dropna()
        # else:
        #     df = pd.read_csv(path,encoding=encoding)
        data_test = cls.pretreat_test(df)
        #分割数据
        return data_test

    #预处理原始数据
    @classmethod
    def pretreat_test(cls, df: pd.DataFrame)->torch.Tensor:
        # 分割数据,整个作为一个数据
        data_segments = np.array([df])
        # 小波变换提取特征
        data_features = extract_dwt_features(data_segments)
        # 归一化,记得看看，是joblib.load('data_scaler.pkl')还是StandardScaler()
        scaler = StandardScaler()
        data_test = scaler.fit_transform(data_features)
        data_test = data_test.reshape((data_test.shape[0], data_test.shape[1], 1))
        print(data_test.shape)
        data_test = torch.tensor(data_test,dtype=torch.float32).permute(0,2,1)
        return data_test


if __name__ == '__main__':
    ef = EEGFileProcessor()
    print("hh")
    print(ef.predict("./datasets/LieWaves/Lie_Sessions/Raw/S1S2.csv"))