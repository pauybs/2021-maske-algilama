import easygui
import numpy as np
from time import time
import cv2
from datetime import datetime
from easygui import *
from MailGonder import MailGonder

while True:
    labels = ["Maskesiz", "Maskeli"]  # Etiketlerin belirlenmesi
    renkler = ["0, 0, 255", "0, 255, 0"]  # Etiketlerin renklerinin belirlenmesi BGR formatında
    renkler = [np.array(renk.split(",")).astype("int") for renk in renkler]  # Renklerin ayrılması int formata çevrilmesi    
    
    text = "Model seçiniz.."
    title = "Model Seç"
    choices = ["YoloV3 mAp = %90", "YoloV3-Tiny mAp = %79", "YoloV4 mAp = %93", "YoloV4-Tiny mAp = %87", "Kapat"]
    modelchoice = easygui.choicebox(text, title, choices) #Model seçim ekranının yapılması

    if modelchoice == choices[0]: #Modelchoice ve choice değerlerine göre modelin seçilmesi
        print(modelchoice + " seçildi.")
        model = cv2.dnn.readNetFromDarknet("Model/yolov3.cfg", "Model/yolov3_last.weights")
    elif modelchoice == choices[1]:
        print(modelchoice + " seçildi.")
        model = cv2.dnn.readNetFromDarknet("Model/yolov3-tiny.cfg", "Model/yolov3-tiny_last.weights")
    elif modelchoice == choices[2]:
        print(modelchoice + " seçildi.")
        model = cv2.dnn.readNetFromDarknet("Model/yolov4.cfg", "Model/yolov4_last.weights")
    elif modelchoice == choices[3]:
        print(modelchoice + " seçildi.")
        model = cv2.dnn.readNetFromDarknet("Model/yolov4-tiny.cfg", "Model/yolov4-tiny_last.weights")
    else:
        print("Kapatılıyor...")
        break


    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    layers = model.getLayerNames() #Modelin içindeki layerlerin alınması
    layers = [layers[i[0] - 1] for i in model.getUnconnectedOutLayers()] #Layerlerin içinden çıktı verecek layerlerin alınması

    text = "Hangisini kullanmak istersiniz?"
    title = "Kullanım tercihi"
    choices = ["Resim", "Video", "Kamera", "Model seçimine geri dön"]
    rvkchoice = easygui.choicebox(text, title, choices) #Resim, video, kamera seçim ekranının yapılması

    if rvkchoice == choices[0]:#rvkchoice ile choice seçimine göre hangisinin seçileceğinin belirlenmesi
        easygui.msgbox(msg="Lütfen resim seçiniz..", title="Bilgi", ok_button="Tamam")
        path = easygui.fileopenbox(title="Resim seçiniz.", filetypes=["*.jpeg", "*.jpg", "*.png"])
        if not path: #path(dosya) seçimediğinde yapılacak işlem
            ynbox = easygui.ynbox("Devam etmek istiyor musunuz?", "", ("Evet","Hayır"))
            if ynbox == True:
                print("Devam ediliyor...")
                continue
            else:
                print("Kapatılıyor...")
                break

        resim = cv2.imread(path) #path(dosyadan) alınan verinin okunup resim değerine eşitlenmesi
        width, height = 1080, 720 #resize için resimin en ve boyunun belirlenmesi
        resize = (width, height)
        resim = cv2.resize(resim, resize, interpolation=cv2.INTER_AREA) #resim dosyasının resize yapılması
        resim_boy, resim_en = resim.shape[0], resim.shape[1] #resmin yeni en ve boyunun değişkene aktarılması

        blob = cv2.dnn.blobFromImage(resim, 1 / 255.0, (832, 832), swapRB=True, crop=False) #resmin blob formata(4 boyutlu tensöre) çevrilmesi
        model.setInput(blob) #blob formatın modele verilmesi
        baslangic = time() #ne kadar sürede tespit edebildiğini öğrenmek için zamanın başlangıcı
        layerOutputs = model.forward(layers) #
        bitis = time()#tespit süresi için zamanın bitisi

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:

                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.2:
                    box = detection[0:4] * np.array([resim_en, resim_boy, resim_en, resim_boy])
                    (box_mx, box_my, box_en, box_boy) = box.astype("int")

                    bas_x = int(box_mx - (box_en / 2))
                    bas_y = int(box_my - (box_boy / 2))

                    boxes.append([bas_x, bas_y, int(box_en), int(box_boy)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                renk = [int(c) for c in renkler[classIDs[i]]]
                cv2.rectangle(resim, (x, y), (x + w, y + h), renk, 1)
                text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                cv2.putText(resim, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, renk, 1)

        resim = cv2.copyMakeBorder(resim, 0, 100, 0, 0, cv2.BORDER_CONSTANT) #resmin altında bir alan oluşturulması
        filtered_classids = np.take(classIDs, idxs) #
        maskesiz = (filtered_classids == 0).sum()  # toplam maskeli insan sayısı
        maskeli = (filtered_classids == 1).sum()  # toplam maskesiz insan sayısı
        toplam = maskeli + maskesiz

        # Sadece maskesiz olanların sayısının ekrana basılması
        text = "Maskesiz: {}".format(maskesiz)
        cv2.putText(resim, text, (0, 750), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [0, 0, 255], 2)

        # Sadece maskeli olanların sayısının ekrana basılması
        text = "Maskeli: {}".format(maskeli)
        cv2.putText(resim, text, (175, 750), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [0, 255, 0], 2)

        # Tespit edilen tüm objelerin ekrana basılması
        text = "Toplam: {}".format(toplam)
        cv2.putText(resim, text, (325, 750), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255, 255, 255], 2)

        # YOLO nun tespit etme süresinin ekrana basılması
        text = "Tespit Suresi {:.3f} MS".format(bitis - baslangic)
        cv2.putText(resim, text, (0, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255, 255, 255], 2)

        # Maskeli-Maskesiz oranı
        oran = (maskeli / toplam) * 100
        text = "Oran: "
        cv2.putText(resim, text, (800, 750), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255, 255, 255], 2)
        if oran == 0:
            text = "0"
            cv2.putText(resim, text, (900, 750), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255, 255, 255], 2)
        else:
            text = "{:.0f}%".format(oran)
            cv2.putText(resim, text, (900, 750), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255, 255, 255], 2)

        # Maskeli-Maskesiz sayısına göre durumun belirlenmesi
        text = "Durum: "
        cv2.putText(resim, text, (800, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255, 255, 255], 2)

        if oran >= 0.1 and maskesiz >= 3:
            text = "Tehlikeli"
            cv2.putText(resim, text, (900, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [0, 0, 255], 2)

        elif oran != 0 and oran >= 50:
            text = "Riskli"
            cv2.putText(resim, text, (900, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [0, 255, 255], 2)

        else:
            text = "Guvenli"
            cv2.putText(resim, text, (900, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [0, 255, 0], 2)

        cv2.imshow("Image", resim)
        key = cv2.waitKey(0)

        if key == ord('s'):
            an = datetime.now()
            filename = "Output/ımage/" + an.strftime('%Y-%m-%d-%H.%M.%S') + str("-result.jpg")
            path = easygui.filesavebox(title="Kaydet", default=filename)
            cv2.imwrite(path, resim)
            print("Kaydedildi..")
            cv2.destroyAllWindows()
        else:
            print("Kapatılıyor..")
            cv2.destroyAllWindows()


    elif rvkchoice == choices[1] or rvkchoice == choices[2]:
        if rvkchoice == choices[1]:
            easygui.msgbox(msg="Lütfen video seçiniz...", title="Bilgi", ok_button="Tamam")
            path = easygui.fileopenbox(title="Video seçiniz.", filetypes=["*.mp4", "*.avi"])
            if not path:
                ynbox = easygui.ynbox("Devam etmek istiyor musunuz?", "", ("Evet", "Hayır"))
                if ynbox == True:
                    print("Devam ediliyor...")
                    continue
                else:
                    print("Kapatılıyor")
                    break
            cap = cv2.VideoCapture(path)
        elif rvkchoice == choices[2]:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        width, height = 1080, 720
        resize = (width, height)

        fps_baslangic = datetime.now()
        fps = 0
        toplam_frame = 0

        an = datetime.now()
        kayitadi = "Output/Video/" + an.strftime('%Y-%m-%d-%H.%M.%S') + str("-result.avi")
        out = cv2.VideoWriter(kayitadi, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Görüntü kaynağı bulunamadı...")
                cap.release()
                cv2.destroyAllWindows()
                continue

            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            frame_boy, frame_en = frame.shape[0], frame.shape[1]

            toplam_frame += 1
            fps_bitis = datetime.now()
            fpsfark = fps_bitis - fps_baslangic
            if fpsfark.seconds == 0:
                fps = 0.0
            else:
                fps = (toplam_frame / fpsfark.seconds)

            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            model.setInput(blob)
            baslangic = time()
            layerOutputs = model.forward(layers)
            bitis = time()

            boxes = []
            confidences = []
            classIDs = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    if confidence > 0.20:
                        box = detection[0:4] * np.array([frame_en, frame_boy, frame_en, frame_boy])
                        (box_mx, box_my, box_en, box_boy) = box.astype("int")

                        bas_x = int(box_mx - (box_en / 2))
                        bas_y = int(box_my - (box_boy / 2))

                        boxes.append([bas_x, bas_y, int(box_en), int(box_boy)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            if len(idxs) > 0:
                for i in idxs.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    renk = [int(c) for c in renkler[classIDs[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), renk, 1)
                    text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, renk, 1)


            out.write(frame)
            # && # Alta gelecek sayaçlar için 100px boşluk olutşurulması
            frame = cv2.copyMakeBorder(frame, 0, 100, 0, 0, cv2.BORDER_CONSTANT)
            # Ekrana basılacak maskeli, maskesiz, toplam sayılarının bulunması
            filtered_classids = np.take(classIDs, idxs)
            maskesiz = (filtered_classids == 0).sum()
            maskeli = (filtered_classids == 1).sum()
            toplam = maskeli + maskesiz


            # Sadece maskesiz olanların sayısının ekrana basılması
            text = "Maskesiz: {}".format(maskesiz)
            cv2.putText(frame, text, (0, 750), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [0, 0, 255], 2)

            # Sadece maskeli olanların sayısının ekrana basılması
            text = "Maskeli: {}".format(maskeli)
            cv2.putText(frame, text, (175, 750), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [0, 255, 0], 2)

            # Tespit edilen tüm objelerin ekrana basılması
            text = "Toplam: {}".format(toplam)
            cv2.putText(frame, text, (325, 750), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255, 255, 255], 2)

            # YOLO nun tespit etme süresinin ekrana basılması
            text = "Tespit Suresi {:.3f} MS".format(bitis - baslangic)
            cv2.putText(frame, text, (0, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255, 255, 255], 2)

            # FPS
            text = "FPS: {:.1f}".format(fps)
            cv2.putText(frame, text, (400, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255, 255, 255], 2)

            # Maskeli-Maskesiz oranı
            if maskeli == 0 and toplam == 0:
                oran = 0
            else:
                oran = (maskeli / toplam) * 100
            text = "Oran: "
            cv2.putText(frame, text, (800, 750), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255, 255, 255], 2)
            text = "{:.0f}%".format(oran)
            cv2.putText(frame, text, (900, 750), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255, 255, 255], 2)

            # Maskeli-Maskesiz sayısına göre durumun belirlenmesi
            text = "Durum: "
            cv2.putText(frame, text, (800, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255, 255, 255], 2)

            if oran >= 0.1 and maskesiz >= 5:
                text = "Tehlikeli"
                cv2.putText(frame, text, (900, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [0, 0, 255], 2)
                if rvkchoice == choices[2]:
                    mesaj = "Bilgilendirme \n"
                    mesaj+= "Durum: {} \n".format(text)
                    mesaj+= "Maskeli: {} \n".format(maskeli)
                    mesaj+= "Maskesiz: {} \n".format(maskesiz)
                    mesaj+="Bilgilendirmenin zamani: "+an.strftime('%Y-%m-%d %H:%M:%S %Z')
                    MailGonder(mesaj)

            elif oran != 0 and oran >= 50:
                text = "Riskli"
                cv2.putText(frame, text, (900, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [0, 255, 255], 2)

            else:
                text = "Guvenli"
                cv2.putText(frame, text, (900, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [0, 255, 0], 2)
                

            if 1 > 0:
                cv2.imshow("frame", frame)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    an = datetime.now()
                    filename = "Output/Image/" + an.strftime('%Y-%m-%d-%H.%M.%S') + str("-result.jpg")
                    path = easygui.filesavebox(title="Kaydet", default=filename)
                    cv2.imwrite(path, frame)
                    print("Kaydedildi..")
                    cap.release()
                    out.release()
                    cv2.destroyAllWindows()
                    break
                elif key == 27:
                    print("Kapatılıyor..")
                    cap.release()
                    out.release()
                    cv2.destroyAllWindows()
                    break

    elif rvkchoice == choices[3]:
        print("Geri gidiliyor..")
        continue
    elif rvkchoice is None:
        print("Kapatılıyor..")
        exit()
