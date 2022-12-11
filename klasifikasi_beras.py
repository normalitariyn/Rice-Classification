from streamlit_option_menu import option_menu
import plotly.express as px
import streamlit as st
import hydralit_components as hc
import pandas as pd
import numpy as np
import altair as alt
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

#make it look nice from the start
st.set_page_config(layout='wide',initial_sidebar_state='collapsed',)

# specify the primary menu definition
with st.sidebar:
        over_theme = {'txc_inactive': '#F5F5DC','menu_background':'#8FBC8F'}
        st.write("""<h3 style = "text-align: center;"><img src="https://lh3.googleusercontent.com/a/AEdFTp6HhVmu4OOAqfreQyWuUrrQWCJd0AvMnBjbw1JpKg=s192-c-rg-br100" width="90" height="90">
        <br><br> NORMALITA EKA ARIYANTI <p>200411100084 <br>Universitas Trunojoyo Madura <br>litashirinka@gmail.com</p></h3>""",unsafe_allow_html=True)
st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)

menu_data = [
    {'label':"Home"},
    {'label':"Description of Data"},
    {'label':"Dataset"},
    {'label':"Prepocessing"},
    {'label':"Modelling"},
    {'label':"Implementation"},
]

over_theme = {'txc_inactive': '#F5F5DC','menu_background':'#8FBC8F'}
menu_id = hc.nav_bar(menu_definition=menu_data, override_theme=over_theme)

if menu_id == "Home":
    st.write("""<h1 style="text-align:center;"> Aplikasi Klasifikasi Beras </h1>""", unsafe_allow_html=True)
    st.write("""<p style="text-align:center;">Aplikasi klasifikasi beras merupakan sebuah aplikasi yang digunakan untuk melakukan pengklasifikasian beras.
    <br>Aplikasi ini mengklasifikasikan beras menjadi dua jenis, yaitu beras jasmine (1) dan beras gonen(0)
    <br><br><img src="https://media.beritagar.id/2017-07/shutterstock-255077293_1501047169.jpg" width="500" height="300">
    </p>""",unsafe_allow_html=True)
    
elif menu_id == "Description of Data":
   st.header("Description of Data")
   st.image("https://ds393qgzrxwzn.cloudfront.net/resize/m600x500/cat1/img/images/0/Db9rleGntu.jpg")
   expander0 = st.expander("Pengertian Data Klasifikasi Beras")
   expander0.write("""Data klasifikasi beras (rice classification data) merupakan data yang digunakan untuk mengklasifikasikan beras. Data ini mengklasifikasikan
   beras menjadi dua jenis, yaitu beras jasmine (1) dan beras gonen(0)""")
   expander = st.expander("Pengertian Beras")
   expander.write("""Beras merupakan bulir gabah yang sudah dikupas kulitnya dan bagian ini
   sudah dapat dimasak serta di konsumsi yang melalui proses penggilingan dan
   penyosohan. Beras termasuk bagian dari struktur tumbuhan biji-bijian yang terdapat pada tanaman padi yang
   tersusun atas aleuron, endosperma, dan embrio (mata meras) yang dijadikan sebagai sumber arti pangan bagi masyarakat Indonesia dalam kesehariannya.
   Beras memiliki beragam jenis. Pada website ini, beras akan diklasifikasikan menjadi dua jenis yaitu beras jasmine dan beras gonen.
   1. Beras Jasmine

      Beras jasmine atau beras melati adalah jenis beras yang berasal dari Thailand. Masyarakat setempat menyebutnya dengan Thai Hom Mali Rice.
      Varietas ini ditemukan oleh petani lokal pada tahun 1945 dan telah dikembangkan sebagai varietas unggul. Salah satu ciri khas yang paling mencolok pada beras melati adalah wangi nasi matang yang menyerupai wangi bunga melati. Hal ini karena terdapat gen aromatik
      amino aldehida pada beras tersebut. Ditambah lagi, tekstur yang lembut dan halus saat dimasak menjadi keunggulan lain beras jasmine.
      
   2. Beras Gonen
   
      Beras Gonen merupakan beras hasil panen lokal daerah Turki. Beras ini digunakan sebagai bahan utama masakan favorit kaya akan vitamin B yang
      memiliki kandungan mangan dan fosfor yang tinggi memiliki karakteristik minyak yang rendah.""")
   
   expander1 = st.expander('Fitur Dataset')
   expander1.write("""Dataset ini terdiri dari 18185 record data dan 10 fitur.
   Fitur-fitur dari dataset ini antara lain :
   1.  Area             - float
   2.  MajorAxisLength  - float
   3.  MinorAxisLength  - float
   4.  Eccentricity     - float
   5.  ConvexArea       - float
   6.  EquivDiameter    - float
   7.  Extent           - float
   8.  Perimeter        - float
   9.  Roundness        - float
   10. AspectRation     - float""")
   expander2 = st.expander('Sumber Data Set')
   expander2.write (""" Dataset yang digunakan bersumber dari kaggle. Berikut link dataset yang digunakan
   https://www.kaggle.com/datasets/seymasa/rice-dataset-gonenjasmine""")

elif menu_id == "Dataset":
   st.header("Dataset Klasifikasi Beras")
   dataset = "https://raw.githubusercontent.com/normalitariyn/dataset/main/riceClassification.csv"
   df = pd.read_csv(dataset)
   st.dataframe(df)

elif menu_id == "Prepocessing":
   st.header("Prepocessing")

   dataset = "https://raw.githubusercontent.com/normalitariyn/dataset/main/riceClassification.csv"
   df = pd.read_csv(dataset)
   
   #definifi variabel x dan y
   X = df.drop(columns=['Class','id'])
   y = df['Class'].values
   df_min = X.min()
   df_max = X.max()

   #normalisasi nilai x
   scaler = MinMaxScaler()
   scaled = scaler.fit_transform(X)
   features_names = X.columns.copy()
   scaled_features = pd.DataFrame(scaled, columns=features_names)


   #tampilkan normalisasi data tanpa label
   st.subheader("Normalisasi Data")
   st.dataframe(scaled_features)

   st.subheader("Target Data")
   dumies = pd.get_dummies(df.Class).columns.values.tolist()
   dumies = np.array(dumies)

   labels = pd.DataFrame({
       "Jasmine"    : [dumies[1]],
       "Gonen"      : [dumies[0]]
   })
   st.dataframe(labels)
    
    
elif menu_id == "Modelling":
   st.header("Modelling")
   
   #import dataset
   dataset = "https://raw.githubusercontent.com/normalitariyn/dataset/main/riceClassification.csv"
   df = pd.read_csv(dataset)

   #definifi variabel x dan y
   X = df.drop(columns=['Class','id'])
   y = df['Class'].values
   df_min = X.min()
   df_max = X.max()

   #normalisasi nilai x
   scaler = MinMaxScaler()
   scaled = scaler.fit_transform(X)
   features_names = X.columns.copy()
   scaled_features = pd.DataFrame(scaled, columns=features_names)

   #split data
   training, test = train_test_split(scaled_features, test_size=0.2, random_state=1)
   training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)

   with st.form("my_form"):
       st.write("Pilih pemodelan data :")
       DecTree = st.checkbox("Decision Tree")
       KNN = st.checkbox("K-Nearest Neighboor")
       GNBayes = st.checkbox("Gaussian Naive Bayes")
       submit=st.form_submit_button("Check")
       
       #create model
       
       #Decision Tree
       dt = DecisionTreeClassifier(random_state=1)
       dt.fit(training, training_label)
       dt_pred = dt.predict(test)

       #Gaussian Naive Bayes
       gnb = GaussianNB()
       gnb = gnb.fit(training, training_label)
       probas = gnb.predict_proba(test)
       probas = probas[:,1]
       probas = probas.round()
       

       #K-Nearest Neighbor
       k=7
       knn=KNeighborsClassifier(n_neighbors=k)
       knn.fit(training,training_label)
       knn_predict=knn.predict(test)
       

       #show akurasi

       #Decision Tree
       dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))
                          
       #Gaussian Naive Bayes
       gnb_akurasi = round(100*accuracy_score(test_label,probas))
                          
       #K-Nearest Neighbor
       knn_akurasi = round(100*accuracy_score(test_label,knn_predict))
                
       if submit:
            if DecTree:
                st.info("Akurasi dari Decision Tree: {0:0.2f}".format(dt_akurasi))
            if GNBayes:
                st.info("Akurasi dari Gaussian Naive Bayes: {0:0.2f}".format(gnb_akurasi))
            if KNN:
                st.info("Akurasi dari KNN: {0:0.2f}".format(knn_akurasi))            
       grafik = st.form_submit_button("Grafik akurasi seluruh permodelan")
       if grafik:
            model = pd.DataFrame({
                "Akurasi" : [dt_akurasi, gnb_akurasi, knn_akurasi],
                "Model" : ["Decision Tree", "Gaussian Naive Bayes", "KNN"],
                })
            bar_chart = px.line(model, 
                    x='Model', 
                    y='Akurasi',
                    text='Akurasi',
                    color_discrete_sequence =['teal']*len(model),
                    template= 'plotly_white')
            bar_chart   
    
elif menu_id == "Implementation":

   st.header("Implementation")

   #import dataset
   dataset = "https://raw.githubusercontent.com/normalitariyn/dataset/main/riceClassification.csv"
   df = pd.read_csv(dataset)

   #definifi variabel x dan y
   X = df.drop(columns=['Class','id'])
   y = df['Class'].values
   df_min = X.min()
   df_max = X.max()

   #normalisasi nilai x
   scaler = MinMaxScaler()
   scaled = scaler.fit_transform(X)
   features_names = X.columns.copy()
   scaled_features = pd.DataFrame(scaled, columns=features_names)

   #split data
   training, test = train_test_split(scaled_features, test_size=0.2, random_state=1)
   training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)
       
   #Decision Tree
   dt = DecisionTreeClassifier(random_state=1)
   dt.fit(training, training_label)
   dt_pred = dt.predict(test)

   #Gaussian Naive Bayes
   gnb = GaussianNB()
   gnb = gnb.fit(training, training_label)
   probas = gnb.predict_proba(test)
   probas = probas[:,1]
   probas = probas.round()
   

   #K-Nearest Neighbor
   k=7
   knn=KNeighborsClassifier(n_neighbors=k)
   knn.fit(training,training_label)
   knn_predict=knn.predict(test)
   

   #show akurasi

   #Decision Tree
   dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))
                      
   #Gaussian Naive Bayes
   gnb_akurasi = round(100*accuracy_score(test_label,probas))
                      
   #K-Nearest Neighbor
   knn_akurasi = round(100*accuracy_score(test_label,knn_predict))

   
   with st.form('inputUser'):
       st.subheader("Input Data Rice Classification")
       Area = st.number_input('Area')
       MajorAxisLength = st.number_input('Major Axis Length')
       MinorAxisLength = st.number_input('Minor Axis Length')
       Eccentricity = st.number_input('Eccentricity')
       ConvexArea = st.number_input('Convex Area')
       EquivDiameter = st.number_input('Equivalen Diameter')
       Extent = st.number_input('Extent')
       Perimeter = st.number_input('Perimeter')
       Roundness = st.number_input('Roundness')
       AspectRation = st.number_input('Aspect Ration')
       
       
       hasilPred=st.form_submit_button("Submit")
       if hasilPred:
            inputs = np.array([
                 Area,
                 MajorAxisLength,
                 MinorAxisLength,
                 Eccentricity,
                 ConvexArea,
                 EquivDiameter,
                 Extent,
                 Perimeter,
                 Roundness,
                 AspectRation
            ])
            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min)/(df_max - df_min))
            input_norm = np.array(input_norm).reshape(1,-1)

            #if dt_akurasi > gnb_akurasi > knn_akurasi:
                 #mod == dt
                 #st.write("model dt")
                 ##st.write(dt_prediction)
            #if gnb_akurasi > knn_akurasi > dt_akurasi:
                 #mod = gnb
                 #st.write("model gnb")
                 ##st.write(gnb_prediction)
            #if knn_akurasi > dt_akurasi > gnb_akurasi:
                 #mod = knn
                 #st.write("model knn")
                 ##st.write(knn_prediction)
            
            a = [dt_akurasi,gnb_akurasi, knn_akurasi]
            mod = max(a)
            if mod == dt_akurasi:
                 st.subheader("Model")
                 st.info("Model yang digunakan adalah Decision Tree dengan akurasi: {0:0.2f}".format(dt_akurasi))
                 modelling = dt
            elif mod == gnb_akurasi:
                 st.subheader("Model")
                 st.info("Model yang digunakan adalah Gaussian Naive Bayes dengan akurasi: {0:0.2f}".format(gnb_akurasi))
                 modelling = gnb
            elif mod == knn_akurasi:
                 st.subheader("Model")
                 st.info("Model yang digunakan adalah K-Nearest Neighbor dengan akurasi: {0:0.2f}".format(knn_akurasi))  
                 modelling = knn
            
                 
            input_pred =  modelling.predict(input_norm)

            st.subheader("Hasil Prediksi")
            if input_pred == 1:
                 st.info("Jasmine")
            else:
                 st.success("Gonen")




