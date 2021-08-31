import cv2
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image, ImageDraw
from PIL import ImagePath 
import numpy as np
import os
import pandas as pd
import json
import tensorflow as tf
import pytesseract
import shutil

os.mkdir("./temp")
os.mkdir("./output")
model = tf.keras.models.load_model('./tablenet')

# given predicted boxes approximate the predicted rectangles
def fil_approx_boxes(img):
    cv2.imwrite("temp/test.jpeg",img)
    img = cv2.imread("temp/test.jpeg",0)
    img = cv2.medianBlur(img,5)
    img = cv2.GaussianBlur(img,(13,13),0)
    img = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)[1]

    _, threshold = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY_INV)
    contours,_ = cv2.findContours(threshold, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if x==0 or y==0:
            continue 
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),-1)
        
    img = cv2.GaussianBlur(img,(13,13),0)
    img = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)[1]
    return img

# Save data into respective CSV Files
def save_to_csv(csv_name,data):
  delim = " "
  if data.find(",") != -1:
    delim = ","
  elif data.find("|") != -1:
    delim = "|"

  data_arr = data.split("\n")
  data_arr = [arr for arr in data_arr if len(arr.strip()) != 0]
  with open(csv_name+".csv",'w') as file:
    for line in data_arr:
      line = line.replace(delim,",")
      file.write(line+"\n")
    file.close()

#  Given masked image, Save both tables and extract text from each
def extract_text(img_path="temp/final_masked.jpeg"):
  img = cv2.imread(img_path,0)
  org_img = img
  img = cv2.GaussianBlur(img,(13,13),0)
  img = cv2.threshold(img, 0,255, cv2.THRESH_BINARY)[1]
  kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

  _, threshold = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY_INV)
  contours,_ = cv2.findContours(threshold, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
  idx = 1
  for cnt in contours:
    file_name = "output/Table_"+str(idx)
    x,y,w,h = cv2.boundingRect(cnt)
    if x==0 or y==0 or w*h < 20000:
        continue

    roi = org_img[y:y+h, x:x+w]
    roi = cv2.filter2D(roi, -1, kernel)
    roi = cv2.resize(roi, (int(w*1.25),int(h*1.25)), interpolation = cv2.INTER_AREA)
    data = pytesseract.image_to_string(roi,config='--psm 6',lang='eng')
    cv2.imwrite(file_name+".jpeg",roi) 
    save_to_csv(file_name,data)
    idx += 1

# predict table and column masks and display
def predict_table_masks(img):
  res1, res2 = model.predict(np.array([img]))
  res1 =  np.expand_dims(np.argmax(res1[0], axis=-1), axis=-1)
  res2 = np.expand_dims(np.argmax(res2[0], axis=-1), axis=-1)
  pred_col = np.squeeze(np.where(res1==1,255,0))
  pred_table = np.squeeze(np.where(res2==1,255,0))
  return fil_approx_boxes(pred_table),fil_approx_boxes(pred_col)

# Predict masks and extract text
def predict_and_extract(img_path):
  for file_name in os.listdir("output"):
    os.remove("output/"+file_name)

  image = tf.io.read_file(img_path)
  org_image = tf.image.decode_image(image, channels=3)
  h,w = org_image.shape[0],org_image.shape[1]
  image = tf.image.resize(org_image, [800, 800])
  pred_table, pred_col = predict_table_masks(image)
  tab = np.where(pred_table == 0,0,1)
  mask = np.expand_dims(tab,axis=2)
  mask = np.concatenate((mask,mask,mask),axis=2)
  cv2.imwrite("temp/mask.jpeg",mask)

  mask = cv2.resize(cv2.imread("temp/mask.jpeg"), (w,h), interpolation = cv2.INTER_AREA)
  masked_img= org_image.numpy() * mask
  cv2.imwrite("temp/final_masked.jpeg",masked_img)
  extract_text()
  shutil.make_archive('output', 'zip', "output/")


  
