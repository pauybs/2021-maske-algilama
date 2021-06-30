<p align="center">
  <img width="520" height="300" src="https://github.com/hheren/2021-maske-algilama/blob/main/DemoOutput/demoresim.png">
</p>


# Medikal(Cerrahi) Maske Algılama 
2020 yılında başlayarak pandemiye dönüşen koronavirüs salgını günden güne can almaya devam ederken ülkelerin aldığı kararlar bunların önüne geçmek istemekte. Ülkelerin aldığı kararlardan biri de medikal(cerrahi) maske kullanımı.  Ülkeden ülkeye değişen kullanım şartları kimi ülkede zorunlu iken kimi ülkede insanın kendi insiyatifine bırakılmış. Bizim burada amaçladığımız zorunlu olan uygulamanın denetlenebilirliğini arttırmak. 

# Veriseti
Toplam **920** adet **JPG** formatında resim kullanıldı.
Eğitim için kullanılan: **800**
Test için kullanılan: **120**


Veriler Maskeli ve Maskesiz diye etiketlenerek iki gruba ayrılmıştır. 
Görseller genel olarak Kaggle ve Github üzerinden alınmıştır. 

Datasetin weight dosyaları boyut sınırları nedeninle [Google Drive](https://drive.google.com/drive/folders/1gmMSbe5Dof4hZvgvcdK8aiKHZf0CjGsV?usp=sharing) hesabına yüklenmiştir. 
# Kullanılan Kütüphaneler
+ Python
+ Opencv(GPU)
+ YoloV4(Modelin yaratılması)
+ EasyGUI
+ Numpy

