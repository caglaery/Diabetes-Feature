# Pima Indians Diabetes Prediction

## Proje Açıklaması
Bu proje, Pima Kızılderili kadınlarının sağlık ölçümlerine dayanarak diyabet hastası olup olmadıklarını tahmin etmek için bir makine öğrenmesi modeli geliştirmeyi amaçlamaktadır. Model geliştirilmeden önce, veri analizi ve özellik mühendisliği adımları gerçekleştirilecektir.

## Veri Seti
Veri seti, ABD Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüsü tarafından toplanmıştır ve Arizona Eyaleti'nin Phoenix şehrinde yaşayan 21 yaş ve üzeri Pima Indian kadınlarını içermektedir.

### **Değişkenler**
| Değişken Adı                 | Açıklama |
|------------------------------|----------|
| **Pregnancies**              | Hamilelik sayısı |
| **Glucose**                  | 2 saatlik plazma glikoz konsantrasyonu |
| **BloodPressure**            | Kan basıncı (mm Hg) |
| **SkinThickness**            | Cilt kalınlığı (mm) |
| **Insulin**                  | 2 saatlik serum insülini (mu U/ml) |
| **DiabetesPedigreeFunction** | Ailede diyabet olasılığı |
| **BMI**                      | Vücut kitle indeksi (kg/m²) |
| **Age**                      | Yaş (yıl) |
| **Outcome**                  | Diyabet olup olmadığı (0: Sağlıklı, 1: Diyabet) |

## **Görev 1: Keşifçi Veri Analizi (EDA)**
### **Adım 1: Genel Resmi İnceleme**
- Veri seti hakkında genel bilgiler edinilir.
- Eksik ve aykırı değerlerin kontrolü yapılır.
- Değişkenlerin özet istatistikleri incelenir.

### **Adım 2: Numerik ve Kategorik Değişkenleri Belirleme**
- Sayısal (numerik) ve kategorik değişkenlerin ayrımı yapılır.

### **Adım 3: Değişken Analizleri**
- Sayısal değişkenlerin dağılımları incelenir.
- Kategorik değişkenlerin sınıf dağılımları incelenir.

### **Adım 4: Hedef Değişken Analizi**
- Kategorik değişkenlere göre hedef değişkenin ortalaması hesaplanır.
- Hedef değişkene göre numerik değişkenlerin ortalamaları hesaplanır.

### **Adım 5: Aykırı Gözlem Analizi**
- Boxplot kullanılarak aykırı değerler belirlenir.
- Gerekirse aykırı değerleri düzeltme işlemleri yapılır.

### **Adım 6: Eksik Gözlem Analizi**
- Eksik değerler tespit edilir.
- Eksik değerleri doldurmak için uygun yöntemler belirlenir.

### **Adım 7: Korelasyon Analizi**
- Değişkenler arasındaki ilişki korelasyon matrisi ile incelenir.
- Yüksek korelasyon gösteren değişkenler belirlenir.

---

## **Görev 2: Feature Engineering (Özellik Mühendisliği)**
### **Adım 1: Eksik ve Aykırı Değer İşlemleri**
- Glucose, Insulin, BloodPressure, SkinThickness ve BMI değişkenlerinde 0 değeri olan gözlemler eksik olarak işaretlenir.
- Eksik değerler medyan veya ortalama ile doldurulur.

### **Adım 2: Yeni Değişkenler Oluşturma**
- **Yaşa Göre Kategorik Gruplama**: Yaş değişkenine göre genç, orta yaşlı ve yaşlı grupları oluşturulur.
- **BMI Kategorik Hale Getirilir**: Zayıf, normal, fazla kilolu ve obez kategorileri oluşturulur.
- **Glukoz Seviyesine Göre Risk Skoru**: Düşük, orta ve yüksek risk seviyeleri tanımlanır.

### **Adım 3: Encoding İşlemleri**
- Kategorik değişkenler `Label Encoding` veya `One-Hot Encoding` yöntemi ile sayısal hale getirilir.

### **Adım 4: Numerik Değişkenleri Standartlaştırma**
- Verilerin ölçeklendirilmesi için `StandardScaler` veya `MinMaxScaler` uygulanır.

### **Adım 5: Modelleme**
- Veriyi eğitim ve test setlerine ayırma (train-test split).
- **Makine Öğrenmesi Algoritmalarının Kullanımı:**
  - **Lojistik Regresyon**
  - **Random Forest**
  - **XGBoost**
- **Model Performans Değerlendirme:**
  - Accuracy (Doğruluk Oranı)
  - Precision, Recall, F1-score
  - ROC-AUC Değeri
- **Model Tuningi**: Hyperparameter tuning işlemi uygulanır.

---

## **Sonuç ve Öneriler**
- Eksik ve aykırı değerlerin giderilmesi modelin performansını artırmıştır.
- Glucose ve BMI değişkenleri diyabet tahmininde önemli bir rol oynamaktadır.
- Model performansını artırmak için daha fazla veri toplanabilir veya farklı algoritmalar denenebilir.

---

## **Kullanım**
Bu proje, Python programlama dili ve aşağıdaki kütüphaneler kullanılarak geliştirilmiştir:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
```
Projeyi çalıştırmak için, `Pima Indians Diabetes Database` veri setini indirip uygun bir çalışma ortamında (`Jupyter Notebook`, `Google Colab`, `VS Code`) çalıştırabilirsiniz.

---

## **Kaynaklar**
- [Pima Indians Diabetes Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

