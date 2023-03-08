'''
kakizaki.py
概要:先輩のコードを参考に、手の関節(landpoint)のxyzを読み取るプログラムを作成する
Author:Kakizki
Date: 2022-10-10
'''


#!/usr/bin/env python3
from re import T
import matplotlib.pyplot as plt
import numpy as np
import cv2
from HandTrackerRenderer import HandTrackerRenderer
import argparse
import time
import os
from tqdm import tqdm

def savehand(name, dataframe):
    #name="madarame_by_fi"
    #print( name + "を保存 ")

    #一度ｎumpyをlistに　もどす
    data_list = dataframe.tolist()
    data_index = 0
    new_data = []
    #[x,y,z][x2,y2,z2]の状態にする
    for i in range(int(len(data_list)/3)):
        x = data_list[data_index]
        data_index += 1
        y = data_list[data_index]
        data_index += 1
        z = data_list[data_index]
        data_index += 1
        new_point = [x,y,z]
        new_data.append(new_point)
    new_data = np.array(new_data)
    np.savetxt(name,new_data,delimiter=',')
        
    '''
    for i in range(len(dataframe)):
        filename = name + "_" + str(i) + ".csv"
        np.savetxt(filepath + filename, dataframe[i])
        #np.savetxt("./data/" + filename, dataframe[i])
    '''

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--edge', action="store_true",
                    help="Use Edge mode (postprocessing runs on the device)")
parser_tracker = parser.add_argument_group("Tracker arguments")
parser_tracker.add_argument('-i', '--input', type=str, 
                    help="Path to video or image file to use as input (if not specified, use OAK color camera)")
parser_tracker.add_argument("--pd_model", type=str,
                    help="Path to a blob file for palm detection model")
parser_tracker.add_argument('--no_lm', action="store_true", 
                    help="Only the palm detection model is run (no hand landmark model)")
parser_tracker.add_argument("--lm_model", type=str,
                    help="Landmark model 'full', 'lite', 'sparse' or path to a blob file")
parser_tracker.add_argument('--use_world_landmarks', action="store_true", 
                    help="Fetch landmark 3D coordinates in meter")
parser_tracker.add_argument('-s', '--solo', action="store_true", 
                    help="Solo mode: detect one hand max. If not used, detect 2 hands max (Duo mode)")                    
parser_tracker.add_argument('-xyz', "--xyz", action="store_true", 
                    help="Enable spatial location measure of palm centers")
parser_tracker.add_argument('-g', '--gesture', action="store_true", 
                    help="Enable gesture recognition")
parser_tracker.add_argument('-c', '--crop', action="store_true", 
                    help="Center crop frames to a square shape")
parser_tracker.add_argument('-f', '--internal_fps', type=int, 
                    help="Fps of internal color camera. Too high value lower NN fps (default= depends on the model)")                    
parser_tracker.add_argument("-r", "--resolution", choices=['full', 'ultra'], default='full',
                    help="Sensor resolution: 'full' (1920x1080) or 'ultra' (3840x2160) (default=%(default)s)")
parser_tracker.add_argument('--internal_frame_height', type=int,                                                                                 
                    help="Internal color camera frame height in pixels")   
parser_tracker.add_argument("-lh", "--use_last_handedness", action="store_true",
                    help="Use last inferred handedness. Otherwise use handedness average (more robust)")                            
parser_tracker.add_argument('--single_hand_tolerance_thresh', type=int, default=10,
                    help="(Duo mode only) Number of frames after only one hand is detected before calling palm detection (default=%(default)s)")
parser_tracker.add_argument('--dont_force_same_image', action="store_true",
                    help="(Edge Duo mode only) Don't force the use the same image when inferring the landmarks of the 2 hands (slower but skeleton less shifted)")
parser_tracker.add_argument('-lmt', '--lm_nb_threads', type=int, choices=[1,2], default=2, 
                    help="Number of the landmark model inference threads (default=%(default)i)")  
parser_tracker.add_argument('-t', '--trace', type=int, nargs="?", const=1, default=0, 
                    help="Print some debug infos. The type of info depends on the optional argument.")                
parser_renderer = parser.add_argument_group("Renderer arguments")
parser_renderer.add_argument('-o', '--output', 
                    help="Path to output video file")
args = parser.parse_args()
dargs = vars(args)
tracker_args = {a:dargs[a] for a in ['pd_model', 'lm_model', 'internal_fps', 'internal_frame_height'] if dargs[a] is not None}

if args.edge:
    from HandTrackerEdge import HandTracker
    tracker_args['use_same_image'] = not args.dont_force_same_image
else:
    from HandTracker import HandTracker


tracker = HandTracker(
        input_src=args.input, 
        use_lm= not args.no_lm, 
        use_world_landmarks=args.use_world_landmarks,
        use_gesture=True,#use_gesture=args.gesture,
        xyz=True,#xyz=args.xyz,
        solo=True,#solo=args.solo,
        crop=args.crop,
        resolution=args.resolution,
        stats=True,
        trace=args.trace,
        use_handedness_average=not args.use_last_handedness,
        single_hand_tolerance_thresh=args.single_hand_tolerance_thresh,
        lm_nb_threads=args.lm_nb_threads,
        **tracker_args
        )

renderer = HandTrackerRenderer(
        tracker=tracker,
        output=args.output)

inp = input('名前を入力してください')
hand_type = input("何番の手形から記録し始めますか？")
hand_type = int(hand_type)
#debug

#data/入力したディレクトリ名/データ保存先
filepath="../data/"+str(inp)+"/"
if not os.path.exists(filepath): #フォルダが存在しないなら新規作成
    os.makedirs(filepath)
if not os.path.exists(filepath+"/landmark/"): #フォルダが存在しないなら新規作成
    os.makedirs(filepath+"/landmark/")
if not os.path.exists(filepath+"/img/"): #フォルダが存在しないなら新規作成
    os.makedirs(filepath+"img")
dataframe = []
pre_z = np.zeros(21)
handdata = []
flag = 0
n = 0#現在の50音に対して何回目の保存なのか

while True:

    frame, hands, bag, depth, coordinates = tracker.next_frame()

    if frame is None:break
    #このあと、renderer.drawに通すとランドマークが追加された画像になるため、コピーしておく
        # Draw hands
    frame2 = renderer.draw(frame, hands, bag)

    key = renderer.waitKey(delay=1)

    #動画をプロット
    cv2.imshow("Hand tracking", frame2)

    '''
    if n==10:
        n=0
        hand_type+=1
        print("次の50音になります。next->"+str(hand_type))
        if hand_type==41:
            print("全種類撮り終わりました。")
            break
    '''
    if hand_type == 47:
        print("全種類の撮影が終わりました")
        break
        
    #ここで実行中の様子が人差し指の座標で確認できるようにする
    #print("x:" + str(hands.world_landmarks))
    #print("y:" + str(hands.landmarks[8][1]))
    #print("z:" + str((hands.norm_landmarks[:,2:3] * hands.rect_w_a  / 0.4).astype(np.int)))

    if key == 27 or key == ord('q'):
        handdata = np.array(handdata)
        dataframe.append(handdata)
        break
    elif key == 13: #enter key
        #スペースキーを押したときに、next_frame()を実行して　データを取ってくるとすると、手の状態をリアルタイム表示できなくなるので、エンターを押したときに保存するというふうにする
        if hand_type > 41:#もし次のターゲットが動きのある50音だったら
            #一度キー入力を意味のないものにする
            key = 0
            n = 0
            while True:
                key = renderer.waitKey(delay=1)
                #もう1度enterが入力されたらキャプチャー終了
                if key == 13:
                    break
                #100集めるループの時も新しいフレームをとってこないとおかしくなる
                frame, hands, bag, depth, coordinates = tracker.next_frame()
                

                '''重すぎて画像の更新はできない
                if i % 7 == 0:#何回もimshowをすると重くなりそうなので、7の倍数のタイミングで画面を更新してみる
                    frame2 = renderer.draw(frame, hands, bag)
                    cv2.imshow("Hand tracking", frame2)
                '''
                #早すぎるとすごく重くなるので、止める
                #time.sleep(0.05)

                capture_data = np.array([])
                for h in hands:
                    for i in range(21): #21 is the num of joints
                        joint_data = np.array([])
                        if not np.any(coordinates == 0.0):
                            flag=1
                        joint_data = np.hstack((joint_data,h.landmarks[i][0]))
                        joint_data = np.hstack((joint_data,h.landmarks[i][1]))
                        if coordinates[i][2]==0.0:
                            joint_data = np.hstack((joint_data,pre_z[i]))
                        else:
                            joint_data = np.hstack((joint_data,coordinates[i][2]))
                            pre_z[i] = coordinates[i][2]
                        
                        #print(joint_data)
                        joint_data = np.array(joint_data)
                        capture_data = np.concatenate([capture_data,joint_data])

                    #print(capture_data)
                    #pre_hand = handdata
                    #データが正しく取れなかった場合にflagが立つ。その場合は前のフレームのデータを採用する
                
                #ここまででてのデータはcapture_dataに格納されたはずなので、これを保存する
                saveName_landmark = '{}/{}_{}_{}'.format(filepath+"landmark",inp,hand_type,n)
                saveName_img = '{}/{}_{}_{}'.format(filepath+"img",inp,hand_type,n)
                saveName_img_landmark = '{}/{}_{}_{}'.format(filepath+"img-landmark",inp,hand_type,n)
                savehand(saveName_landmark+".csv",capture_data)
                cv2.imwrite(saveName_img+".jpg",frame)
                #データを大量に保存する関係で、画像を保存したくないのでランドマーク付きの画像は保存しないことにする
                #cv2.imwrite(saveName_img_landmark+".jpg",frame2)
                n+=1#現在保存した画像枚数をインクリメント
        
        else:#動きのない50音の場合　
            
            n = 0
            
            for i in tqdm(range(100)):#画像を100枚集める
                #100枚集めるループの時も新しいフレームをとってこないとおかしくなる
                frame, hands, bag, depth, coordinates = tracker.next_frame()
                

                '''重すぎて画像の更新はできない
                if i % 7 == 0:#何回もimshowをすると重くなりそうなので、7の倍数のタイミングで画面を更新してみる
                    frame2 = renderer.draw(frame, hands, bag)
                    cv2.imshow("Hand tracking", frame2)
                '''
                #早すぎるとすごく重くなるので、止める
                #time.sleep(0.05)

                capture_data = np.array([])
                for h in hands:
                    for i in range(21): #21 is the num of joints
                        joint_data = np.array([])
                        if not np.any(coordinates == 0.0):
                            flag=1
                        joint_data = np.hstack((joint_data,h.landmarks[i][0]))
                        joint_data = np.hstack((joint_data,h.landmarks[i][1]))
                        if coordinates[i][2]==0.0:
                            joint_data = np.hstack((joint_data,pre_z[i]))
                        else:
                            joint_data = np.hstack((joint_data,coordinates[i][2]))
                            pre_z[i] = coordinates[i][2]
                        
                        #print(joint_data)
                        joint_data = np.array(joint_data)
                        capture_data = np.concatenate([capture_data,joint_data])

                    #print(capture_data)
                    #pre_hand = handdata
                    #データが正しく取れなかった場合にflagが立つ。その場合は前のフレームのデータを採用する
                
                #ここまででてのデータはcapture_dataに格納されたはずなので、これを保存する
                saveName_landmark = '{}/{}_{}_{}'.format(filepath+"landmark",inp,hand_type,n)
                saveName_img = '{}/{}_{}_{}'.format(filepath+"img",inp,hand_type,n)
                saveName_img_landmark = '{}/{}_{}_{}'.format(filepath+"img-landmark",inp,hand_type,n)
                savehand(saveName_landmark+".csv",capture_data)
                cv2.imwrite(saveName_img+".jpg",frame)
                #データを大量に保存する関係で、画像を保存したくないのでランドマーク付きの画像は保存しないことにする
                #cv2.imwrite(saveName_img_landmark+".jpg",frame2)
                n+=1#現在保存した画像枚数をインクリメント
        hand_type += 1#次の50音へ
        print("次の50音は " + str(hand_type) + "です")
    elif key == 122:#zが入力されたら一つ戻る
        hand_type -= 1
        print("一つ戻りました。")
        print("50音 : "+str(hand_type))
        print(str(n) + "枚目")
        

    


renderer.exit()
tracker.exit()

#dataframe = np.array(dataframe)

'''
#Save hands data
name = "kaki-test"
savehand(name, dataframe)
print(dataframe.shape)
print("-")
'''