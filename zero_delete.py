#depthcameraの座標に対し標準化を行う
import os
from tqdm import tqdm
import pandas as pd
from sklearn import preprocessing

#mpのcsvファイルのリストを取得し、そのリストに含まれないdcのcoordinateファイルを弾いた新たなファイル群を作成する

dirName = input('標準化したいのは誰のデータですか？: ')

srcDir1 = "./data/"+dirName+"/alin_num_coordinates/"#dcの座標が入ったディレクトリ
wrtDir = "./data/"+dirName+"/zero_deleted_landmark/"#ファイル数を揃えて書き込むディレクトリ

try:#書き込むためのディレクトリを作成
    os.makedirs(wrtDir,exist_ok = True)
except FileExistsError:
    pass

srcFiles1 = os.listdir(srcDir1)#ファイル名のリストを取得

srcFiles1.sort()

del_num = 0
for f in tqdm(srcFiles1):
    #ここでdcの座標データがmpに含まれているのかを一個一個確かめていく
    #もし含まれていないならそれを保存せず、含まれていたなら保存する
    try:
        df = pd.read_csv(srcDir1+f,header=None)
    except:
        print("例外発生" + f)

    #df_contain_0に0のデータが含まれていたらTrueになるようにする
    df_contain_0 = (df == 0)
    #0が含まれているデータなら飛ばす
    if df_contain_0.sum().sum() >= 1:
        print("0が含まれていました")
        continue

    # CSVとして出力
    df.to_csv(wrtDir+f,header=None,index = False)

print("無効となったファイル数は"+str(del_num)+"個でした")