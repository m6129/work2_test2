import io # обязательные библиотеки для stremlit
import streamlit as st # # обязательные библиотеки для stremlit
from PIL import Image # библиотека для загрузки изображений
import torch
from torchvision import transforms
import urllib

model = torch.hub.load('nicolalandro/ntsnet-cub200', 'ntsnet', pretrained=True,
                       **{'topN': 6, 'device':'cpu', 'num_classes': 200})


    
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
    model = torch.hub.load('nicolalandro/ntsnet-cub200', 'ntsnet', pretrained=True, **{'topN': 6, 'device':'cpu', 'num_classes': 200})
    model.eval()

    url = 'https://raw.githubusercontent.com/nicolalandro/ntsnet-cub200/master/images/nts-net.png'
    img = Image.open(urllib.request.urlopen(url))
    scaled_img = transform_test(img)
    torch_images = scaled_img.unsqueeze(0)

    with torch.no_grad():
        top_n_coordinates, concat_out, raw_logits, concat_logits, part_logits, top_n_index, top_n_prob = model(torch_images)

        _, predict = torch.max(concat_logits, 1)
         pred_id = predict.item()
        print('bird class:', model.bird_classes[pred_id])
