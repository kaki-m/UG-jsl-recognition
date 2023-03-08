'''
画像を読み込んで、mediapipeに通した結果の手の座標を出力する
'''
import cv2
import mediapipe as mp
import os
import csv
import numpy as np
from natsort import natsorted

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

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For static images:

#static_image_mode=True
#ax_num_hands=検出する手の数　デフォルト２
#n_detection_confidence=[0.0,1.0]検出が成功したとみなされるための、手の検出モデルからの最小信頼値 デフォルト0.5


hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.01)
data=[]

folder_name = input("./data/@@@/imgの画像を処理する")

try:
    os.makedirs("./data/"+ folder_name +"/mediapipe_csv", exist_ok=True)
except FileExistsError:
    pass
try:
    os.makedirs("./data/" + folder_name + "/mediapipe_img", exist_ok=True)
except FileExistsError:
    pass

File_from = "./data/"+folder_name+"/img/"
file_list=os.listdir(File_from)
file_list = natsorted(file_list)
fault_img_num = 0

for idx, file in enumerate(file_list):
  # Read an image, flip it around y-axis for correct handedness output (see
  # above).
  image = cv2.flip(cv2.imread(File_from+file), 1)
  # Convert the BGR image to RGB before processing.
  try:
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  except:
    print("エラーが出てしまう画像なのでcontinueします")
    fault_img_num += 1
    continue

  # Print handedness and draw hand landmarks on the image.
  #print('Handedness:', results.multi_handedness)
  if not results.multi_hand_landmarks:
    print("エラーが出てしまう画像なのでcontinueします")
    fault_img_num += 1
    continue
  image_hight, image_width, _ = image.shape
  annotated_image = image.copy()
  #identify_japanese_syllabry_flag = (file.split("/")[2].split("_")[2])

  for hand_landmarks in results.multi_hand_landmarks:
    capture_data = np.array([])
    for i in hand_landmarks.landmark: #21 is the num of joints
        joint_data = np.array([])
        joint_data = np.hstack((joint_data,i.x))
        joint_data = np.hstack((joint_data,i.y))
        joint_data = np.hstack((joint_data,i.z))
        
        joint_data = np.array(joint_data)
        capture_data = np.concatenate([capture_data,joint_data])
    file_name_without_ext = file.split('.', 1)[0]
    savehand("./data/"+ folder_name +"/mediapipe_csv/" + file_name_without_ext+".csv",capture_data)

    mp_drawing.draw_landmarks(
        annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
  
  cv2.imwrite("./data/" + folder_name + "/mediapipe_img/" + file , cv2.flip(annotated_image, 1))
hands.close()

print("失敗した画像の枚数は" + str(fault_img_num) + "枚でした")
