import io # обязательные библиотеки для stremlit
import streamlit as st # # обязательные библиотеки для stremlit
from PIL import Image # библиотека для загрузки изображений


    
def load_image():
    uploaded_file = st.file_uploader(label='Загрузите пожалуйста изображение') # загрузчик файлов
    if uploaded_file is not None: # если пользователь загрузил файл
        image_data = uploaded_file.getvalue() # то мы его читаем
        st.image(image_data) # преобразуем с помощью средств stremlit
        return Image.open(io.BytesIO(image_data))# возвращаем это изображение
    else:
        return None
st.title('Классификация изображений')
img = load_image() # вызываем функцию
#mod = load_model()

result = st.button('Распознать изображение')# вставляем кнопку
st.write('**Успешно3:**')
###пробуем
if result: #после нажатия на которую будет запущен алгоритм...
    st.write('**Результаты распознавания:**')
    #image_to_text(img) #стоит убрать решётку и будет выдавать ошибку(пробовал по разному запускать модель и через функцию - итог один)
