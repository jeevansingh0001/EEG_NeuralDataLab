import pandas as pd

from os import listdir



# for i in range(len(csv_files)):
def extractData(fileName):
    data = pd.read_csv(fileName, skiprows=1)
    # print(data.head())

    data.columns
    columns = ['EEG.Counter', 'EEG.Interpolated',
            'EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 'EEG.P7', 'EEG.O1',
            'EEG.O2', 'EEG.P8', 'EEG.T8', 'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4']

    df = data[columns]
    df.drop([k for k in range(0,124)], axis=0, inplace=True)
    df.drop([k for k in range(124,636)], axis=0, inplace=True)
    df.drop([k for k in range(32000,len(df))], axis=0, inplace=True)

    df = df.reset_index()
    df.drop('index', axis=1, inplace=True)

    df['time_s'] = 0

    count=1
    for j in range(len(df)):
        df['time_s'].iloc[j] = count
        if(df['EEG.Counter'].iloc[j]==127):
            count = count+1

    t = [1]*128*10 + [1]*128*10 + [1]*128*10 + [0]*128*10 + [0]*128*10 + [1]*128*10 + [0]*128*10 + [1]*128*10 + [0]*128*10 + [1]*128*10 + [0]*128*10 + [0]*128*10 + [0]*128*10 + [1]*128*10 + [0]*128*10 + [0]*128*10 + [1]*128*10 + [0]*128*10 + [1]*128*10 + [0]*128*10 + [1]*128*10 + [0]*128*10 + [1]*128*10 + [1]*128*10 + [1]*128*10


    df['target'] = t
    df.head()


    df.drop('time_s', axis=1, inplace=True)

    df.to_csv(fileName + '(prepared_data).csv', index=False)
    print("Data Processed")