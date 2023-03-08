#depthcameraの座標に対し標準化を行う
'''
このプログラムは座標データの標準化を行うプログラムです
一枚一枚のフレームのmax,minで標準化を行ってしまうと、動きのあるジェスチャーを認識する際にフレーム間の比較ができないので
同じhand_type全て合わせたmax,minを使用して計算する
'''
import os
from tqdm import tqdm
import pandas as pd
from sklearn import preprocessing
from natsort import natsorted

data_type = input('depthCamera: 0\nmediaPipe: 1\nどちらのデータですか？: ')
if not(data_type in ['0','1']):
    print("無効な命令です")
    exit()
#mpのcsvファイルのリストを取得し、そのリストに含まれないdcのcoordinateファイルを弾いた新たなファイル群を作成する
dirName = input('~~~~~~~~~~~\nhand_typeごとに標準化し\n0を含むファイルを削除したあと\nファイル名を正しく保存します\n~~~~~~~~~~\n誰のディレクトリに対し実行しますか？: ')

if data_type == '0':
    srcDir = "./data/"+dirName+"/alin_num_coordinates/"#dcの座標が入ったディレクトリ
    wrtDir = "./data/"+dirName+"/preprocessed_coordinates_dc/"#ファイル数を揃えて書き込むディレクトリ
    try:
        os.makedirs(wrtDir, exist_ok=True)
    except FileExistsError:
        pass
elif data_type == '1':
    srcDir = "./data/"+dirName+"/mediapipe_csv/"#dcの座標が入ったディレクトリ
    wrtDir = "./data/"+dirName+"/preprocessed_coordinates_m/"#ファイル数を揃えて書き込むディレクトリ
    try:
        os.makedirs(wrtDir, exist_ok=True)
    except FileExistsError:
        pass

try:#書き込むためのディレクトリを作成
    os.makedirs(wrtDir,exist_ok = True)
except FileExistsError:
    pass

srcFiles = os.listdir(srcDir)#ファイル名のリストを取得
srcFiles = natsorted(srcFiles)

#46あるhand_typeそれぞれのmax,minを保存するための配列
INF = 10000000.0
maxs = [[0,0,0]for i in range(47)]
mins = [[INF,INF,INF] for i in range(47)]
#hand_typeを格納する変数の宣言
hand_type = 0

for f in tqdm(srcFiles):
    #このループでhand_typeごとのmax,minを計算していく
    #csvデータをdfに入れる
    try:
        df = pd.read_csv(srcDir+f,header=None)
    except:
        print("例外発生" + f)

    df_contain_0 = (df == 0)
    #0が含まれているデータなら飛ばす
    if df_contain_0.sum().sum() >= 1:
        continue
    
    #hand_typeを見る
    hand_type = int(f.split("_")[1])

    #x座標のmaxが現時点より大きいなら、max書き換え
    if maxs[hand_type][0] < df[0].values.max():
        maxs[hand_type][0] = df[0].values.max()
    if maxs[hand_type][1] < df[1].values.max():
        maxs[hand_type][1] = df[1].values.max()
    if maxs[hand_type][2] < df[2].values.max():
        maxs[hand_type][2] = df[2].values.max()
    
    if mins[hand_type][0] > df[0].values.min():
        mins[hand_type][0] = df[0].values.min()
    if mins[hand_type][1] > df[1].values.min():
        mins[hand_type][1] = df[1].values.min()
    if mins[hand_type][2] > df[2].values.min():
        mins[hand_type][2] = df[2].values.min()
    #df_contain_0に0のデータが含まれていたらTrueになるようにする


current_type = 0#次のhand_typeになったか確認するために保持する
file_num = 0
del_num = 0
for f in tqdm(srcFiles):
    #さっきまで変えていたhand_typeと違うものになったらファイル数を0に戻す
    if current_type != int(f.split("_")[1]):
        current_type = int(f.split("_")[1])
        file_num = 0
    #ここでdcの座標データがmpに含まれているのかを一個一個確かめていく
    #もし含まれていないならそれを保存せず、含まれていたなら保存する
    try:
        df = pd.read_csv(srcDir+f,header=None)
    except:
        print("例外発生" + f)

    #df_contain_0に0のデータが含まれていたらTrueになるようにする
    df_contain_0 = (df == 0)
    #0が含まれているデータなら飛ばす
    if df_contain_0.sum().sum() >= 1:
        print("0が含まれていました")
        del_num += 1
        continue

    #読み込んだdfにMinMaxScalerをして標準化してから書き込む
    df[0]= (df[0] - mins[current_type][0]) / (maxs[current_type][0] - mins[current_type][0])
    df[1]= (df[1] - mins[current_type][1]) / (maxs[current_type][1] - mins[current_type][1])
    df[2]= (df[2] - mins[current_type][2]) / (maxs[current_type][2] - mins[current_type][2])

    #ファイル名を変更する
    saveName = dirName + "_" + str(current_type) + "_" + str(file_num) + ".csv"
    # CSVとして出力
    df.to_csv(wrtDir+saveName,header=None,index = False)
    file_num += 1
print("無効となったファイル数は"+str(del_num)+"個でした")