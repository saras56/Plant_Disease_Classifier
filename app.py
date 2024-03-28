import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# Model Prediction
class_names =['Early_blight','Late_blight', 'Healthy']
def model_prediction(test_image):
    model = load_model('bestmodel_CNN.h5')
    img = tf.keras.preprocessing.image.load_img(test_image,target_size=(256,256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array=img_array/255 #scaling is important 
    img_array = tf.expand_dims(img_array,0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence  = round(100*(np.max(predictions[0])),2)
    return predicted_class,confidence

#Sidebar
st.sidebar.title('Dashboard')
app_mode = st.sidebar.selectbox('Seelct Page', ['Home','About','Disease Recognition'])

#Home Page
if (app_mode == 'Home'):
    st.header('Potato Plant Disease Recognition System')
    st.markdown("""
    Welcome to the Potato Plant Disease Recognition System! 🌿🔍
    
    Early detection and accurate diagnosis of leaf diseases are crucial for effective disease management and prevention of its spread. 
    The traditional methods of disease diagnosis rely on visual inspection by experts which is time-consuming and can be prone to errors. 
    In case of potato plants, the diseases earlt blight and late blight are the most frequent. 
    As the treatment for these diseases are different it is important to identify them accurately and early so as to minimize the economical loss by farmers. 
    The Potato Plant Disease Recognition System classifies potato plant leaf diseases into 3 categories : Late Blight, Early Blight and Healthy(no disease).
    Upload an image of a potato plant, and the system will analyze it to detect any signs of diseases.

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** The system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.



    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of the Potato Plant Disease Recognition System!

    """)
    #About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                The dataset used for this model is taken from Plant Village Dataset available on Kaggle. There were a total of 2152 images belonging to three classes namely Late Blight ,Early Blight ,Healthy.
                The split is done in such a way that 70% of the data goes to training, 10% for validation and remaining 20% for test.  
                The test data is completely unseen. There are 1506 train images, 215 validation images an 431 test images.
                

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Prediction")
        predicted_class,confidence = model_prediction(test_image)
        st.success("Model is Predicting it's an {}".format(predicted_class))
        st.success("Confidence in prediction is {} %".format(confidence))
    