import pandas as pd
import logging
import os
import json
import calc

def load_aihub_envs():
    ''' aihub 환경데이터 로딩 '''
    df = []
    for i in range(1, 7):
        if i == 2 or i == 3:
            df.append(load_aihub_priva(i))
        else:
            df.append(load_aihub_env(i))
    df = pd.concat(df, sort=False)
    return df


def load_aihub_envs_old():
    ''' aihub 환경데이터 로딩 '''
    df = []
    for i in range(1, 7):
        if i == 2 or i == 3:
            df.append(load_aihub_priva(i))
        else:
            df.append(load_aihub_env(i))
    df = pd.concat(df, sort=False)

    df['dpt'] = df.apply(lambda x: calc.dpt(x['dtemp'], x['humidity']), axis=1)

    df_day = df.between_time('06:30', '18:30')


    df_night = df.between_time('12:30', '13:40')
    df_am = df.between_time('00:00', '11:59')
    df_pm = df.between_time('12:00', '23:59')
    df_hightemp = df[df['dtemp'] >= 30]
    df_lowtemp = df[df['dtemp'] <= 12]

    df_day_grouped = df_day.groupby(['farmIdx', df.index.get_level_values('time').strftime('%Y-%m-%d')])
    df_day_mean = df_day_grouped.mean().add_prefix('mean_')
    df_day_max = df_day_grouped.max().add_prefix('max_')
    df_day_min = df_day_grouped.min().add_prefix('min_')
    df_day_std = df_day_grouped.std().add_prefix('std_')
    df_daily = pd.concat([df_day_mean, df_day_max, df_day_min, df_day_std], axis=1)

    print(df_daily.head(10))

    return df_daily


def load_aihub_env(farmIdx):
    ''' 1, 4, 5, 6번 농가 환경데이터 '''
    base_dir = 'dataset/aihub_tomato/Training/env/'
    if farmIdx == 6:
        file_path = f'{base_dir}tom{farmIdx}/01_(함수율저울,열화상센서,작물환경센서)/MV_header.csv'
    else:
        file_path = f'{base_dir}tom{farmIdx}/01_(함수율저울,열화상센서,작물환경센서)/MV.csv'

    df = pd.read_csv(file_path)
    #df = df[['time', '내부PT100온도센서1번건구', '내부PT100온도센서2번습구', '내부PT100센서를이용한계산습도']]
    df = df[['time', '내부PT100온도센서1번건구', '내부PT100센서를이용한계산습도']]
    df.columns = ['time', 'dtemp', 'humidity']

    df['farmIdx'] = farmIdx
    df['time'] = pd.to_datetime(df['time'])
    #df.set_index(keys=['time'], inplace=True, drop=True)

    return df


def load_aihub_priva(farmIdx):
    ''' 2번, 3번 농가 환경데이터 '''
    base_dir = 'dataset/aihub_tomato/Training/env/'
    if farmIdx == 2:
        file_name = f'토마토_프리바_환경데이터_{farmIdx}번농가'
    elif farmIdx == 3:
        file_name = f'토마토_환경데이터_{farmIdx}번농가'
    file_path = f'{base_dir}tom{farmIdx}/00_(환경제어기, 양액기)/{file_name}.csv'
    df = pd.read_csv(file_path)

    if farmIdx == 2:
        df = df.iloc[:, [0, 6, 9]]
    elif farmIdx == 3:
        df = df.iloc[:, [0, 5, 6]]
    df.columns = ['time', 'humidity', 'dtemp']

    df['farmIdx'] = farmIdx
    df['time'] = pd.to_datetime(df['time'])
    # df['wtemp'] = df.apply(lambda x: calc.wbt(x['dtemp'], x['humidity']), axis=1)
    #df.set_index(keys=['time'], inplace=True, drop=True)

    return df

def load_aihub_growthes():
    #base_dir = 'dataset/aihub_tomato'
    #self.load_aihub_plant_height()
    df1 = load_aihub_growth('a1.생장길이')
    df3 = load_aihub_growth('a3.줄기두께')
    df4 = load_aihub_growth('a4.엽장엽폭')

    df = pd.merge(df1, df3, on=['farmId', 'number', 'week', 'date'])
    df = pd.merge(df, df4, on=['farmId', 'number', 'week', 'date'])
    print(df.head(20))
    #train
    pass

def load_aihub_growth(item_name):
    # 1. 줄기두께 json 파일 리스트 구해서
    #dir = f'dataset/aihub_tomato/Training/{item_name}'
    dir = f'dataset/aihub_paprika/Training/{item_name}'
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

    # 2. json 파일을 하나씩 열어서 annotation된 값을 가져 온다
    objs = []
    for f in files:
        file = os.path.join(dir, f)
        with open(file) as fin:
            json_obj = json.load(fin)
        
        try:
            farmId = json_obj['file_attributes']['farmId']
            number = json_obj['file_attributes']['number']
            week = json_obj['file_attributes']['week']
            date = json_obj['file_attributes']['date']

            obj = {'farmId': farmId, 'number': number, 'week': week, 'date': date}

            if item_name == 'a1.생장길이':
                plantHeight = json_obj['growth_indicators']['plantHeight']
                #weeklyGrowth = json_obj['growth_indicators']['weeklyGrowth']
                obj['plantHeight'] = plantHeight
            elif item_name == 'a3.줄기두께':
                stemDiameter = json_obj['growth_indicators']['stemDiameter']
                obj['stemDiameter'] = stemDiameter
            elif item_name == 'a4.엽장엽폭':
                leafLength = json_obj['growth_indicators']['leafLength']
                leafWidth = json_obj['growth_indicators']['leafWidth']
                obj['leafLength'] = leafLength
                obj['leafWidth'] = leafWidth

            #print(farmId, number, week, date, plantHeight, weeklyGrowth)

            objs.append(obj)

        except:
            pass # passthrough

        if farmId == 'tom4' and number == '077' and week == '46':
            print(obj)
            print(json_obj)

    df = pd.DataFrame(objs)
    print(df.head(20))
    print(df.count())
    df = df.drop_duplicates(subset=['farmId', 'number', 'date'])

    df = df.groupby(['farmId', 'number', 'week', 'date']).mean()

    print(df.head(20))
    print(df.count())

    return df