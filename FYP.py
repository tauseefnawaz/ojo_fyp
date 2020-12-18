import numpy as np

import cv2

model = cv2.dnn.readNet('/content/gdrive/My Drive/yolov3/yolov3_training_final.weights',
                        '/content/gdrive/My Drive/yolov3/yolov3_testing.cfg')

classes = []
with open('/content/gdrive/My Drive/yolov3/classes.txt','r') as f:
  classes = f.read().splitlines()

for file_name in test_files:
  img = cv2.imread(file_name)
  height,width,_ = img.shape

  blob = cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),swapRB=True,crop=False)
  model.setInput(blob)
  output_layer_names = model.getUnconnectedOutLayersNames()
  ########Forward Pass################
  layer_output = model.forward(output_layer_names)

  boxes = []
  confidences = []
  class_ids = []

  for output in layer_output:
    for prediction in output:
      scores = prediction[5:]
      class_id = np.argmax(scores)
      confidence = scores[class_id]
      if(confidence>0.4):
        center_x = int(prediction[0]*width)
        center_y = int(prediction[1]*height)
        w = int(prediction[2]*width)
        h = int(prediction[3]*height)

        x = int(center_x-w/2)
        y = int(center_y-h/2)

        boxes.append([x,y,w,h])
        confidences.append(float(confidence))
        class_ids.append(class_id)

  indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.4)
  colors = np.random.uniform(0,255,size=(len(indexes),3))
  if(len(indexes)!=0):
    for i in indexes.flatten():
      x,y,w,h = boxes[i]
      label = str(classes[class_ids[i]])
      confidence = str(round(confidences[i],2))
      color = colors[i]
      cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
      Output = label+" "+confidence
      print(Output)
      cv2.putText(img,label+" "+confidence,(x,y+20),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
