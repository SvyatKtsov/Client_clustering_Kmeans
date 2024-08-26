import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

df = pd.read_csv("D:\\КНУ\\2 курс\\2 семестр\\Курсова_робота\\Datasets\\New_datasets\\telecommunications_dataset.csv")
df_copy = df.copy()

y_chunk_oneHotEn = pd.get_dummies(df_copy['churn'])
x_state_ = pd.get_dummies(df_copy['state'])
x_state_WV_MN = pd.concat([x_state_['WV'],x_state_['MN']],axis=1)
#x_state_WV_MN['MN'].value_counts()
x_state_WV = x_state_WV_MN['WV']
x_state_MN = x_state_WV_MN['MN']
if __name__ == '__main__':
    print(x_state_WV.value_counts(), '\n', x_state_MN.value_counts())
all_oneHotEn_cols = pd.concat([x_state_WV,x_state_MN,y_chunk_oneHotEn],axis=1)
df_copy = pd.concat([df_copy, all_oneHotEn_cols], axis=1)

y_churn_yes = df_copy['yes']
#print(y_churn_yes.value_counts(),'\n')
x_number_vmail_messages = df_copy['number_vmail_messages']
x_total_intl_calls = df_copy['total_intl_calls']
x_number_customer_service_calls = df_copy['number_customer_service_calls']
#x_number_vmail_messages.value_counts()
x_state_WestVirginia = df_copy['WV']
x_state_Minnesota = df_copy['MN']


#########################
all_needed_cols_forLoopPC = list(df.columns)[7:17] #+18
all_needed_cols_forLoopPC.append(list(df.columns)[18])

if __name__ == "__main__":
    print("This will only be executed when Data_exploration.py is run")
