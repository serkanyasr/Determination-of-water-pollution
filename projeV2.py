################################################################################
######## ÇÖZELTİDE KLOR MİKTARINI BELİRLEME VE TAHMİN ETME #####################
################################################################################
# BY:           SERKAN YAŞAR
# DATE:         MART 2022
################################################################################

import pandas as pd 
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import math
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn import metrics
################################################################################
#### RESİM İSİMLERİNİ ÇEKME  ###################################################
yol=os.getcwd()
yol=os.path.join(yol,"resimler")
dosya_yolu= os.listdir("resimler")
################################################################################
G_Values=[] 
B_Values=[] 

def ResimAdı():
    files_name=[]     
    for f in dosya_yolu:
        if f.endswith(".jpeg"):      # resim isimmlerini çekilir. isimler klor miktarına göre adladırılmalıdır 1.2.3  0,1 0,2 

            files_name.append(f)

    return files_name

def ResimAdDüzen(files_name):
    Just_File_name=[]
    for i in files_name:
        a=i[0:i.find(".")]
        Just_File_name.append(int(a))
    return Just_File_name

def Resimİslem(files_name):

    for i in files_name:                                  # for döngüsü ile her bir resimden R değerini alıyoruz
        img= cv2.imread(os.path.join(yol,"{}").format(i))
        print(img)
        img= cv2.resize(img,(800,800))                    # resmi tekrar boyutlnadırma
        # cv2.imshow("{}".format(i),img)                    
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        height, width, _ = img.shape                        # resmin boyutlarını alıp değişkene atıyoruz
        cx = int((width / 2))                               #resmin orta noktasını buluyoruz
        cy = int((height / 2))

        pt1=(cx+100,cy+100)                                 # belirli alandaki RGB kodlarını okuması için alan belirliyoruz Karenin köşegeninin noktaları
        pt2=(cx-100,cy-100)

        crop= img[pt2[1]:pt1[1],pt2[0]:pt1[0]]              # belirlenen kare alanını orjinal resimden kesiliyor
        
        crop2=crop.copy()
        
        kernel=np.ones((2,2),dtype=np.uint8)                # morfolijik işlem için 2x2 lik kutucuk  oluşturuyoruz

        crop=cv2.morphologyEx(crop,cv2.MORPH_OPEN,kernel)    # morfolojik işlem yapılıyor

        crop=cv2.mean(crop)
        
        crop2=cv2.cvtColor(crop2,cv2.COLOR_BGR2HSV)
        crop2=cv2.morphologyEx(crop2,cv2.MORPH_OPEN,kernel)    # morfolojik işlem yapılıyor
        
        crop2=cv2.mean(crop2)


        b,g,r= int(crop[0]),int(crop[1]),int(crop[2])
            
        b,g,r= int(crop[0]),int(crop[1]),int(crop[2])        
        h,s,v= int(crop2[0]),int(crop2[1]),int(crop2[2])       
        
        a="R:{},G:{},B:{}".format(r,g,b)                      # ekrana yazdırmak için değişkene aktarma
        x="H:{},S:{},V:{}".format(h,s,v)                      # ekrana yazdırmak için değişkene aktarma
        
        cv2.putText(img,a,(25,100),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,0),2)   # R,G,B değerlerini ekrana yazdırma
        cv2.putText(img,x,(25,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,0),2)   # R,G,B değerlerini ekrana yazdırma
        cv2.rectangle(img,pt1,pt2,(255,255,0),thickness=5)

        cv2.imshow("{}".format(i),img)   
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        deger = int((g +b)/2)
        G_Values.append(deger)



    return G_Values 

def ABS_Hesaplama(Files_name, G_Values):
    
    dic={"file":Files_name,"G":G_Values,"ABS_values":np.nan}
    df=pd.DataFrame(dic)
    for i in np.arange(0,len(df["G"])):
        df["ABS_values"][i]=-math.log10(df["G"][i]/df["G"][0])
    df["ABS_values"][0]=-df["ABS_values"][0]
    # print(df.ABS_values)
    ABS_values=df["ABS_values"].to_list()
    ABS_values=np.array(ABS_values)
    df.to_excel("renk2.xlsx")
    return ABS_values
 
def Regresyon(resim_sayısı,ABS_values):
    resim_sayısı=np.array(resim_sayısı)
    resim_Series= pd.Series(resim_sayısı)               # exele kayıt için Series oluşturur
    ABS_series=pd.Series(ABS_values)
    data=dict(Resim_Sayısı=resim_Series,ABS_degerleri=ABS_series)                                 # DataFrame oluşturma
    df=pd.DataFrame(data) 

    plt.scatter(resim_sayısı,ABS_values)              # Resim sayısı- ABS değer grafiği
    plt.title("Resim Sayısı - ABS grafiği")
    plt.xlabel("Resim Saysısı")
    plt.ylabel("ABS değerleri")
    plt.show()

    x = df.Resim_Sayısı
    y_gercek=df.ABS_degerleri              # bağımlı değişken (target, dependent variable)


    # Sabitin eklenmesi
    x = sm.add_constant(x)

    # Modelin çalıştırılması
    model = sm.OLS(y_gercek,x).fit()


    # Modelin yorumlanacağı tablo
    print(model.summary())


    fig, ax = plt.subplots(figsize=(8, 8))      # 800X800 e figure oluşturma
    fig = sm.graphics.plot_ccpr(model,"Resim_Sayısı" ,ax=ax)   # grafikte gösterme değerleri

    plt.title("Resim Sayısı - ABS grafiği")
    plt.xlabel("Resim Saysısı")
    plt.ylabel("ABS değerleri")
    plt.savefig("cıktılar\\grafik.jpeg")
    plt.show()







    # Doğrusal regresyon sınıfı çağırılıyor
    lr = LinearRegression()

    # Model verileri içine alarak uyarlanıyor
    lr.fit(x,y_gercek)

    # uyarlanan model aynı yaşlar için tahmini gelir sağtıyor
    y_pred_sklearn = lr.predict(x)

    # Gerçek değer değerleri ile tahmin edilen değer  değerlerinin birleştirilmesi- karşılaştırması
    karsilastirma = pd.DataFrame({'Gercek_Degerler': y_gercek, 'Tahmin_Degerler': y_pred_sklearn}).sort_index()
    # Gerçek değer ile tahmin edilen değer arasındaki hata
    karsilastirma["tahminleme_hatalari"] = karsilastirma.Gercek_Degerler - karsilastirma.Tahmin_Degerler
    print(karsilastirma)
    katsayi="katsayı ",lr.coef_  # katsayı

    a= katsayi[1]
    a=a.tolist()

    Egim=a[1]


    b="kesim noktası",lr.intercept_ # Kesim noktası
    Hata_oranı=b[1]

    R_2=metrics.r2_score(y_gercek,y_pred_sklearn)                               # R^2 değeri sapma değeri
    print("r2 değeri !!!",metrics.r2_score(y_gercek,y_pred_sklearn))
    print('ortalama kare hatası:', metrics.mean_squared_error(y_gercek, y_pred_sklearn))
    print('Karekök ortalama hata:', np.sqrt(metrics.mean_squared_error(y_gercek, y_pred_sklearn)))


    # print(Hata_oranı,Egim)

    return Hata_oranı,Egim

def VideoFrame(frame):
    frame=cv2.flip(frame,1)                               # gelen Frame i aynaladık Y düzleminde

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    # BGR dan HSV ye çevirme
    height, width, _ = frame.shape                        #yükseklik ve genişlik bilgisini alma
    cx = int((width / 2))                                 # frame in orta noktalarını belirleme
    cy = int((height / 2))
    pt1=(cx+100,cy+100)                                   # 100x100 diktörtgen çizdirmek için kordinatları belirleme
    pt2=(cx-100,cy-100)

    crop= hsv_frame[pt2[1]:pt1[1],pt2[0]:pt1[0]]          # diktörtgen alanı kırpma işlemi
    
    kernel=np.ones((5,5),dtype=np.uint8)                  # morfolojik açılma için gerekli (5x5) boyutunda kutucuk

    crop=cv2.morphologyEx(crop,cv2.MORPH_OPEN,kernel)     # Açılma işlemi oluşan gürültüyü azaltmak için

    crop=cv2.cvtColor(hsv_frame,cv2.COLOR_HSV2BGR)        # Açılmadan sonra oluşan frame in HSV den RGB ye çevirme
    
    crop=cv2.mean(crop)                                   # daha net sonuçlar alınması için dikdörtgen alanndaki piksellerin ortalaması alınaması R,G,B olarak


    b,g,r= int(crop[0]),int(crop[1]),int(crop[2])         # B,G,R değerlerini çekme
    
    a="R:{},G:{},B:{}".format(r,g,b)                    # ekrana yazdırmak için değişkene aktarma
    cv2.putText(frame,a,(25,100),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(255,255,255),2)   # R,G,B değerlerini ekrana yazdırma
    

    Tahmin_edilecek_deger=g
    abs_hesaplama=-math.log(int(Tahmin_edilecek_deger)/G_Values[0])
    conc=(abs_hesaplama-Hata_oranı)/Egim
    conc="kolerasyon degeri:{:.5}".format(conc)


    cv2.putText(frame,conc,(25,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,255),2)   


    # cv2.circle(frame, (cx, cy), 20, (255,255,255), 10)
    cv2.rectangle(frame,pt1,pt2,(255,255,0),thickness=5)  # diktörtgeni ekranda gösterme

    return frame 


File_name=ResimAdı()
    
düzenleniş_File_Name=ResimAdDüzen(File_name)

G_values=Resimİslem(File_name)

    
ABS_values=ABS_Hesaplama(düzenleniş_File_Name, G_values)

Hata_oranı,Egim=Regresyon(düzenleniş_File_Name,ABS_values)

cap= cv2.VideoCapture(0)

while True:
    ret,frame= cap.read()


    frame=VideoFrame(frame) 

    cv2.imshow("video",frame)
    if cv2.waitKey(1)& 0XFF==ord("a"):
        break

        











    