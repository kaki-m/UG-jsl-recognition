#collect_dataの不具合で動きのあるファイル名を正しく保存できていなかったため、それを直すプログラム

import os
from tqdm import tqdm
from natsort import natsorted

dirName = input("ファイル名を修復したいのは誰のデータですか?: ")

#imgとlandmarkのファイル名を修正したい
srcDir = "./data/"+dirName

'''
ファイル名はhand_type42からおかしくなっているので、42が現れたときに
枚数を0にしてファイル名を変えていき、変えるたびに枚数をインクリメント
hand_typeが変わったらまた枚数を0にしてファイル名を変更して行く
'''
#ソースディレクトリの中のファイル名一覧を取得
srcFiles = os.listdir(srcDir+"/img")
#ファイル名をソートするためのライブラリnatsort
srcFiles = natsorted(srcFiles)

#42以降のhand_typeのファイル名を変える
current_type = 42
file_num = 0
for f in tqdm(srcFiles):
    #42以降のhand_typeなら処理をする
    if int(f.split("_")[1]) >= 42:
        #さっきまで変えていたhand_typeと違うものになったらファイル数を0に戻す
        if current_type != int(f.split("_")[1]):
            current_type = int(f.split("_")[1])
            file_num = 0

        #ファイル名を変更する
        print(f.split("_")[0] + "_" + str(current_type) + "_" + str(file_num) + ".jpg")
        file_num += 1