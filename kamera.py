import cv2
import numpy as np

vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    
    frame = cv2.flip(frame,1)
    frame = cv2.resize(frame,(960,720))
    
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    frame_blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), swapRB=True, crop=False)

    labels = ["Maskeli", "Maskesiz"]

    
    renkler = ["0,255,0","0,0,255"]
    renkler = [np.array(renk.split(",")).astype("int") for renk in renkler]
    


    model = cv2.dnn.readNetFromDarknet("Dataset/yolov4-tiny.cfg","Dataset/yolov4-tiny_last.weights")

    layers = model.getLayerNames()
    output_layer = [layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]
    
    model.setInput(frame_blob)
        
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
                
                label = labels[predicted_id]
                bounding_box = object_detection[0:4] * np.array([frame_width,frame_height,frame_width,frame_height])
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
        label = labels[predicted_id]
        confidence = confidences_list[max_class_id]      

                
        son_x = bas_x + kutu_en
        son_y = bas_y + kutu_boy
                
        kutu_renk = renkler[predicted_id]
        kutu_renk = [int(each) for each in kutu_renk]
                
                
        label = "{}: {:.2f}%".format(label, confidence*100)
         
                
        cv2.rectangle(frame, (bas_x,bas_y),(son_x,son_y),kutu_renk,2)
        cv2.rectangle(frame, (bas_x-1,bas_y),(son_x+1,bas_y-30),kutu_renk,-1)
        cv2.putText(frame,label,(bas_x,bas_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.imshow("Video",frame)    
    key = cv2.waitKey(1)    
    if key == ord('n') or key == ord('p'):
        break
cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    