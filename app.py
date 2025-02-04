from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import pandas as pd
import zipfile
import re
import io
import nltk
nltk.download('stopwords')
spanish_stopwords = stopwords.words('spanish')

# Configuración de la app Streamlit
st.title("🗒️ Análisis de Chat de WhatsApp")
st.write("⬇️ Sube un archivo de backup de un chat de WhatsApp para ver las palabras más frecuentes.")

uploaded_file = st.file_uploader("Sube un archivo ZIP con el chat", type=["zip"])

if uploaded_file is not None:
    with zipfile.ZipFile(uploaded_file, "r") as z:
        # Obtener el primer archivo TXT dentro del ZIP
        file_names = [f for f in z.namelist() if f.endswith(".txt")]
        
        if file_names:
            with z.open(file_names[0]) as chat_file:
                chat_text = io.TextIOWrapper(chat_file, encoding="utf-8")
                
                chat = pd.read_csv(chat_text, sep='delimiter', header=None)

                # Renombramos la columna para poder dividirla en dos
                chat.columns = ['chat']
                # Separamos la columna 'chat' en dos columnas: 'timestamp' y 'message'
                chat[['timestamp', 'message']] = chat['chat'].str.extract(r'\[(.*?)\](.*)')

                # Eliminamos la columna 'chat'
                chat = chat.drop('chat', axis=1)

                # Separamos la columna 'message' en dos columnas: 'Usuario' y 'Mensaje'
                grupo_datos = chat['message'].str.split(':', n=1, expand=True)  # Usamos n=1 para dividir solo en el primer ':'

                grupo_datos.columns = ['Usuario', 'Mensaje']
                grupo_datos['Mensaje'] = grupo_datos['Mensaje'].str.apply(lambda x: x.lower().replace("omitido", "eliminado"))

                # Eliminamos filas donde 'Usuario' sea NaN (mensajes del sistema como "María agregó a Juan")
                grupo_datos = grupo_datos.dropna(subset=['Usuario'])

                # Eliminamos espacios en blanco
                grupo_datos['Usuario'] = grupo_datos['Usuario'].str.strip()
                grupo_datos['Mensaje'] = grupo_datos['Mensaje'].str.strip()

                mensajes_por_usuario = grupo_datos.groupby('Usuario')['Mensaje'].count().sort_values(ascending=False)
                
                st.subheader("Cantidad de Mensajes por Usuario")
                fig, ax = plt.subplots()
                sns.barplot(x=mensajes_por_usuario.index, y=mensajes_por_usuario.values, palette='magma', ax=ax)
                
                # Ajustes de formato
                plt.xticks(rotation=90)
                plt.title('Cantidad de mensajes enviados por usuario')
                plt.xlabel('Usuario')
                plt.ylabel('Cantidad de mensajes')
                st.pyplot(fig)
            

                # Realizamos el análisis de palabras más frecuentes
                # Usamos CountVectorizer para extraer las palabras más frecuentes
                vectorizer = CountVectorizer(stop_words=spanish_stopwords, max_features=15)
                
                # Convertimos los mensajes a una matriz de términos
                X = vectorizer.fit_transform(grupo_datos['Mensaje'])
                
                # Sumamos la frecuencia de cada palabra
                word_freq = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out()).sum(axis=0)
                
                # Filtramos palabras de 3 letras o menos
                word_freq = word_freq[word_freq.index.str.len() > 3]

                # Ordenamos de mayor a menor frecuencia
                word_freq = word_freq.sort_values(ascending=False)
                
                # Mostramos el resultado en un gráfico de barras
                st.subheader("Palabras Más Frecuentes en el Chat (Sin Palabras Cortas)")
                st.bar_chart(word_freq)