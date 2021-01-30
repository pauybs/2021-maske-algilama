import cv2
import numpy as np


rsm = cv2.imread("Resim/3.png")

rsm_en = rsm.shape[1]
rsm_boy = rsm.shape[0]


img_blob = cv2.dnn.blobFromImage(rsm, 1/255, (416,416), swapRB=True, crop=False)

etiket = ["Maskeli", "Maskesiz"]

renkler = ["0,255,0","0,0,255"]
renkler = [np.array(renk.split(",")).astype("int") for renk in renkler]


model = cv2.dnn.readNetFromDarknet("Dataset/yolov4-tiny.cfg","Dataset/yolov4-tiny_last.weights")

layers = model.getLayerNames()
output_layer = [layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]

model.setInput(img_blob)

detection_layers = model.forward(output_layer)


ids_list = []
boxes_list = []
confidences_list = []



for detection_layer in detection_layers:
    for object_detection in detection_layer:
        
        scores = object_detection[5:]
        predicted_id = np.argmax(scores)
        confidence = scores[predicted_id]
        
        if confidence > 0.20:
            
            baslik = etiket[predicted_id]
            bounding_box = object_detection[0:4] * np.array([rsm_en,rsm_boy,rsm_en,rsm_boy])
            (kutu_mx, kutu_my, kutu_en, kutu_boy) = bounding_box.astype("int")
            
            bas_x = int(kutu_mx - (kutu_en/2))
            bas_y = int(kutu_my - (kutu_boy/2))
            
            
            
            
            ids_list.append(predicted_id)
            confidences_list.append(float(confidence))
            boxes_list.append([bas_x, bas_y, int(kutu_en), int(kutu_boy)])
            
            
            
max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
     
for max_id in max_ids:
    
    max_class_id = max_id[0]
    box = boxes_list[max_class_id]
    
    bas_x = box[0] 
    bas_y = box[1] 
    kutu_en = box[2] 
    kutu_boy = box[3] 
     
    predicted_id = ids_list[max_class_id]
    baslik = etiket[predicted_id]
    confidence = confidences_list[max_class_id]
  
            
    son_x = bas_x + kutu_en
    son_y = bas_y + kutu_boy
            
    kutu_renk = renkler[predicted_id]
    kutu_renk = [int(each) for each in kutu_renk]
            
            
    baslik = "{}: {:.2f}%".format(baslik, confidence*100)
    print("Tahmin: {}".format(baslik))
     
            
    cv2.rectangle(rsm, (bas_x,bas_y),(son_x,son_y),kutu_renk,1)
    cv2.putText(rsm,baslik,(bas_x+5,bas_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, kutu_renk, 1)

cv2.namedWindow("Resim", cv2.WINDOW_NORMAL)
cv2.imshow("Resim", rsm)     
cv2.waitKey(0)
cv2.destroyAllWindows()