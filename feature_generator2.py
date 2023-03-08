'''
2022/12/25
特徴値生成コード2
２枚の画像間で座標がどのくらい動いたのかを保持する
feature1は連続する2枚の画像だったが、feature2は1->3の一個飛ばしにする
'''
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import math
from natsort import natsorted

mode = input('depthカメラ:0\nmediapipe:1\n')
Name= input("加工したいファイルのディレクトリ名を入力してください:")
#modeが0ならdepthカメラの方を取ってくる
if mode == '0':
    srcdir = 'preprocessed_coordinates_dc'
    wrtdir = 'features2_dc'
elif mode == '1':
    srcdir = 'preprocessed_coordinates_m'
    wrtdir = 'features2_mp'
file=os.listdir("./data/"+Name+"/"+srcdir+"/")#データファイルの一覧を持ってくる
srcFiles = natsorted(file)

#書き込み用ディレクトリを用意する
try:
    os.makedirs("./features/"+wrtdir+"/", exist_ok=True)
except FileExistsError:
    pass

#tqdmは進捗バーを表示する
for f in tqdm(srcFiles):
    data=np.loadtxt("./data/"+Name+'/'+srcdir+'/'+f,dtype='float32',delimiter=',')
    hand_type = f.split('_')[1]#hand_typeをファイル名から取ってくる
    file_num  = int(f.split('_')[2].split('.')[0])
    nextFileName = Name + '_' + hand_type + '_' + str(file_num + 2) + '.csv'
    #次のフレームのファイル名を得るために変数に入れる
    try:
        nextdata = np.loadtxt('./data/' + Name + '/' + srcdir + '/' + nextFileName,dtype='float32',delimiter=',')
    except:#ファイル読み込みに失敗したということは次のhand_typeになったということ
        #print("次のhand_typeに行きます:" + hand_type)
        continue
        
    '''
    data = np.loadtxt("./data/kakizaki/landmark/kakizaki_0_0.csv",dtype='float32',delimiter=',')
    print(data[0][0])
    401.0のように行ごと取ってくる
    '''
    if len(data) == 0:
        print("空のCSVを検出: "+ f)
        continue
    #for angle
    x=[]
    y=[]
    z=[]

    #for dist
    x2=[]
    y2=[]
    z2=[]
    
    #まずはファイルのデータを崩す
    for i in range(21):
        x.append(data[i][0])
        x2.append(data[i][0])
        y.append(data[i][1])
        y2.append(data[i][1])
        z.append(data[i][2])
        z2.append(data[i][2])
    dist=[]
    mark=[]
    dist2=[]
    mark2=[]
    dist3=[]
    mark3=[]
    
    #点と点同士の座標から特徴値を計算していく
    flag = 0
    for i in range(20):
        for j in range(i+1,21):
            if (j-i==1) and i%4!=0  or (i==0 and(j==1 or j==5 or j==9 or j==13 or j==17)):
                continue
            dist.append(math.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2+(z[i]-z[j])**2))#単純な距離
            mark.append(str(i)+" "+str(j))#どことどこの点の特徴か
            
        for j in range(i+1,21):
            if (math.sqrt((x2[j]-x2[i])**2+(y2[j]-y2[i])**2+(z2[j]-z2[i])**2)) == 0:
                print("0で割るケースが発生しました")
                flag = 1
                break
            dist2.append(math.acos((x2[j]-x2[i])/(math.sqrt((x2[j]-x2[i])**2+(y2[j]-y2[i])**2+(z2[j]-z2[i])**2))))
            mark2.append("angleX_"+str(i)+"_"+str(j))
            dist2.append(math.acos((y2[j]-y2[i])/(math.sqrt((x2[j]-x2[i])**2+(y2[j]-y2[i])**2+(z2[j]-z2[i])**2))))
            mark2.append("angleY_"+str(i)+"_"+str(j))
            dist2.append(math.acos((z2[j]-z2[i])/(math.sqrt((x2[j]-x2[i])**2+(y2[j]-y2[i])**2+(z2[j]-z2[i])**2))))
            mark2.append("angleZ_"+str(i)+"_"+str(j))
    
    for i in range(21):#すべてのランドマークに対して一枚目と二枚目の変化(variation)を計算する
        #nextdata - dataが新しい特徴量
        dist3.append(nextdata[i][0] - data[i][0])
        mark3.append("variationX_"+str(i))
        dist3.append(nextdata[i][1] - data[i][1])
        mark3.append("variationY_"+str(i))
        dist3.append(nextdata[i][2] - data[i][2])
        mark3.append("variationZ_"+str(i))

    if flag == 1:#0で割るケースは飛ばす
        continue
    dist=np.array(dist)
    dist2=np.array(dist2)
    dist3 = np.array(dist3)
    df = pd.DataFrame(data=[dist], columns=mark)
    df2 = pd.DataFrame(data=[dist2], columns=mark2)
    df3 = pd.DataFrame(data=[dist3],columns=mark3)
    df = pd.concat([df,df2],axis=1)
    df = pd.concat([df,df3],axis=1)

    df.to_csv("./features/"+wrtdir+'/'+f,index=False)#データの保存
