import io # обязательные библиотеки для stremlit
import streamlit as st # # обязательные библиотеки для stremlit
from PIL import Image # библиотека для загрузки изображений

import torch
import requests
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel


loc = "ydshieh/vit-gpt2-coco-en"

feature_extractor = ViTFeatureExtractor.from_pretrained(loc)
tokenizer = AutoTokenizer.from_pretrained(loc)
model = VisionEncoderDecoderModel.from_pretrained(loc)
model.eval()


def predict(image):

    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

    with torch.no_grad():
        output_ids = model.generate(pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True).sequences

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    return preds


# We will verify our results on an image of cute cats
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
with Image.open(requests.get(url, stream=True).raw) as image:
    preds = predict(image)

print(preds)
#@st.cache(allow_output_mutation=True)
#def load_model():
    #return image-to-text


    
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
if result: #после нажатия на которую будет запущен алгоритм...
    st.write('**Результаты распознавания:**')
    #image_to_text(img) #стоит убрать решётку и будет выдавать ошибку(пробовал по разному запускать модель и через функцию - итог один)
    st.title('Мне кажется модель https://huggingface.co/nlpconnect/vit-gpt2-image-captioning не работает со стремлит')
