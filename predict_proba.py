import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def predict_proba(X_train, Y_train, X_test, base_model, validation_data=None, output_layer="logi"):

    '''
    Data reshape and preprocessing: 
    will convert the images from RGB to BGR, 
    then will zero-center each color channel with respect to the ImageNet dataset, without scaling.
    https://www.tensorflow.org/api_docs/python/tf/keras/applications/imagenet_utils/preprocess_input
    '''
    X_train = preprocess_input(np.asarray(X_train).reshape(-1,224,224,3))
    X_test = preprocess_input(np.asarray(X_test).reshape(-1,224,224,3))
    
    Y_train = np.asarray(Y_train)

    if validation_data != None:
        X_val = preprocess_input(np.asarray(validation_data[0]).reshape(-1,224,224,3))
        Y_val = np.asarray(validation_data[1])

    '''
    Feature Extractor:
    '''

    datagen = ImageDataGenerator()

    base_model.trainable = False # Freeze the parameters for transfer learning
    model = tf.keras.models.Sequential()
    model.add(base_model)
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    '''
    Feature Extracting:
    '''
    
    X_train = model.predict(datagen.flow(X_train, batch_size=32, shuffle=False))
    X_test = model.predict(datagen.flow(X_test, batch_size=32, shuffle=False))

    print('Features extracted!')
    '''
    Output layer:
    '''
    if output_layer == "svm":
        layer = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=False, random_state=42))
        layer.fit(X_train, Y_train)
        y_pred_proba = layer.predict(X_test)

    elif output_layer == "fc":
        X_val = model.predict(datagen.flow(X_val, batch_size=32, shuffle=False))
        
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation="relu"))
        model.add(tf.keras.layers.Dense(256, activation="relu"))
        model.add(tf.keras.layers.Dense(2, activation="softmax"))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])
       
        Y_train = tf.one_hot(Y_train, 2)
        Y_val = tf.one_hot(Y_val, 2)
        
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        history = model.fit(x=X_train, y=Y_train, validation_data=(X_val, Y_val), epochs=10, callbacks=[callback])
        
        y_pred_proba = model.predict(X_test)
    
    return y_pred_proba