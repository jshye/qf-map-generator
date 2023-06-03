import os
from math import floor
import requests
import zipfile
import streamlit as st
import plotly.express as px
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO


@st.cache_resource()
def load_model(model_path):
    if not os.path.exists(model_path):
        model_name = model_path.split('/')[-1]
        model_url = st.secrets[model_name]

        with st.spinner('Downloading model... this may take a while!'):
            response = requests.get(model_url)
            with open(f'{model_name}.zip', 'wb') as f:
                f.write(response.content)

            with zipfile.ZipFile(f'{model_name}.zip') as z:
                z.extractall(model_path)
            
    model = tf.keras.models.load_model(model_path)

    return model


@st.cache_data
def load_image(img_file):
    img = Image.open(img_file)
    st.image(img, caption='Uploaded Image', width=200)
    return img


@st.cache_data
def show_qfmap(qfmap, title):
    fig = px.imshow(qfmap)
    fig.update_layout(title_text=title, title_x=0.35, 
                                margin={'autoexpand': True})
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_coloraxes(colorbar_len=1, 
                                    colorbar_orientation='h', 
                                    colorbar_thickness=15,
                                    colorbar_yanchor='bottom',
                                    colorbar_y=-0.5)
    st.plotly_chart(fig, use_container_width=True)


def preprocess_img(img):
    x = np.array(img)
    x = tf.image.resize([x], [224, 224], method='bicubic')
    x = tf.clip_by_value(x, 0., 255.)
    x = tf.cast(x, 'float32')
    return x


@st.cache_data
def get_bpp(img, img_size=224.):  # workaround
    img_ = tf.keras.utils.array_to_img(img[0])
    with BytesIO() as buffer:
        img_.save(buffer, format='PNG')
        bpp = buffer.getbuffer().nbytes / (img_size * img_size) 
    return bpp


def qf_to_scale(qf):
    qf = int(qf)
    if qf < 50 and qf >= 1:
        scale = floor(5000 / qf)
    elif qf < 100 and qf >= 50:
        scale = 200 - 2 * qf
    else:
        scale = 10  # QF95
    return scale / 100.


def main():
    st.set_page_config(layout='wide')

    with st.sidebar:
        st.title('An Overhead-Free Region-Based JPEG Framework \
                for Task-Driven Image Compression')
        file = st.file_uploader("Upload an Image", type=["jpg", "jpeg"])

    try:
        std_jpeg = load_model('./assets/stdjpeg-scale01')
    except:
        std_jpeg = None

    try:
        model = load_model('./assets/end-to-end')
    except:
        st.text('Failed to load QF Map generator and predictor')

    if file is None:
        st.text('Waiting for image upload...')
    else:
        with st.sidebar:
            img = load_image(file)
            qf = st.slider("Quality Factor for the standard JPEG", 
                           min_value=2, max_value=50, value=8, step=1)
        
        x = preprocess_img(img)

        logits, _, enc_qf, _, dec_qf, _, _, qdct, rgb, _ = model(x)
        preds = tf.keras.applications.imagenet_utils.decode_predictions(logits.numpy(), top=1)
        bpp = get_bpp(rgb.numpy())

        ct1 = st.container()

        with ct1:
            ct1_c1, ct1_c2, ct1_c3 = st.columns((1, 1, 1))

            with ct1_c1:
                st.image(x.numpy()/255., width=250, use_column_width=True, caption='Resized Input Image')

            with ct1_c2:
                st.image(rgb.numpy()/255., width=250, use_column_width=True, caption='Region-Based JPEG')
                st.text(f'[ResNet50] {preds[0][0][1]} {preds[0][0][2]:.2%}')
                st.text(f'{bpp:.4f} BPP')  #TODO: bpp calculation from qdct

        if std_jpeg:
            scale = qf_to_scale(qf)
            std_jpeg.scale.assign(scale)
            
            std_logits, _, std_qdct, std_rgb, _ = std_jpeg(x)
            std_preds = tf.keras.applications.imagenet_utils.decode_predictions(std_logits.numpy(), top=1)
            std_bpp = get_bpp(std_rgb.numpy())

            with ct1_c3:
                st.image(std_rgb.numpy()/255., width=250, use_column_width=True, caption='Standard JPEG')
                st.text(f'[ResNet50] {std_preds[0][0][1]} {std_preds[0][0][2]:.2%}')
                st.text(f'{std_bpp:.4f} BPP')  #TODO: bpp calculation from qdct
        else:
            with ct1_c3:
                st.text('Failed to load standard jpeg')
        
        with st.expander('See QF Maps'):
            ec1, ec2 = st.columns((1,1))

            with ec1:
                st.markdown("<h3 style='text-align: center; color: #E1D9D1;'>Generated QF Maps</h3>", 
                            unsafe_allow_html=True)
                
                ec1_1, ec1_2 = st.columns((1,1))
                
                with ec1_1:
                    show_qfmap(enc_qf[0,:,:,0].numpy(), 'Luma Channel')

                with ec1_2:
                    show_qfmap(enc_qf[0,:,:,1].numpy(), 'Chroma Channel')

            with ec2:
                st.markdown("<h3 style='text-align: center; color: #E1D9D1;'>Predicted QF Maps</h3>", 
                            unsafe_allow_html=True)
                
                ec2_1, ec2_2 = st.columns((1,1))

                with ec2_1:
                    show_qfmap(dec_qf[0,:,:,0].numpy(), 'Luma Channel')

                with ec2_2:
                    show_qfmap(dec_qf[0,:,:,1].numpy(), 'Chroma Channel')


if __name__ == "__main__":
    main()
