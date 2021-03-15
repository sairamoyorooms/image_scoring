from python.scoring_models.scoring_models import *
from tqdm import tqdm
import numpy as np

def aes_predict_model(X_val1, aes_mean_mL):
    print('Uploading Aesthetic model...')
    from keras.layers import Dense, Dropout
    from keras.applications import MobileNet
    from keras.models import Model
    aes_base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    aes_x = Dropout(0.75)(aes_base_model.output)
    aes_x = Dense(10, activation='softmax')(aes_x)
    aes_model = Model(aes_base_model.input, aes_x)
    aes_model.load_weights(aesthetic_model)
    for i in tqdm(range(X_val1.shape[0])):
        aes_out = aes_model.predict(np.expand_dims(X_val1[i], 0))[0]
        a_mean = 1 * aes_out[0] + 2 * aes_out[1] + 3 * aes_out[2] + 4 * aes_out[3] + 5 * aes_out[4] + 6 * \
                 aes_out[
                     5] + 7 * aes_out[6] + 8 * aes_out[7] + 9 * aes_out[8] + 10 * aes_out[9]
        aes_mean_mL.append(a_mean)

def tech_predict_model(X_val1, tech_mean_mL):
    print('Uploading Technical model...')
    from keras.layers import Dense, Dropout
    from keras.applications import MobileNet
    from keras.models import Model
    tech_base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    tech_x = Dropout(0.75)(tech_base_model.output)
    tech_x = Dense(10, activation='softmax')(tech_x)
    tech_model = Model(tech_base_model.input, tech_x)
    tech_model.load_weights(technical_model)
    for i in tqdm(range(X_val1.shape[0])):
        tech_out = tech_model.predict(np.expand_dims(X_val1[i], 0))[0]
        t_mean = 1 * tech_out[0] + 2 * tech_out[1] + 3 * tech_out[2] + 4 * tech_out[3] + 5 * tech_out[4] + 6 * \
                 tech_out[5] + 7 * tech_out[6] + 8 * tech_out[7] + 9 * tech_out[8] + 10 * tech_out[9]
        tech_mean_mL.append(t_mean)

def angle_predict_model(X_val1, angle_prob_mL):
    print('Uploading Angle positioning model...')
    from keras.layers import Dense, Dropout
    from keras.applications import MobileNet
    from keras.models import Model
    ang_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    ang_x = Dropout(0.75)(ang_model.output)
    ang_x = Dense(2, activation='softmax')(ang_x)
    model_angle = Model(ang_model.input, ang_x)
    model_angle.load_weights(angle_model)
    for i in tqdm(range(X_val1.shape[0])):
        ang_out = model_angle.predict(np.expand_dims(X_val1[i], 0))[0]
        angle_prob_mL.append(ang_out[0])

def comp_predict_model(X_val1, comp_prob_mL):
    print('Uploading Composition model...')
    from keras.layers import Dense, Dropout
    from keras.applications import MobileNet
    from keras.models import Model
    comp_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    comp_x = Dropout(0.75)(comp_model.output)
    comp_x = Dense(2, activation='softmax')(comp_x)
    model_comp = Model(comp_model.input, comp_x)
    model_comp.load_weights(composition_model)
    for i in tqdm(range(X_val1.shape[0])):
        comp_out = model_comp.predict(np.expand_dims(X_val1[i], 0))[0]
        comp_prob_mL.append(comp_out[0])

def dist_predict_model(X_val1, dist_prob_mL):
    print('Uploading Distortion model...')
    from keras.layers import Dense, Dropout
    from keras.applications import MobileNet
    from keras.models import Model
    dist_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    dist_x = Dropout(0.75)(dist_model.output)
    dist_x = Dense(2, activation='softmax')(dist_x)
    model_dist = Model(dist_model.input, dist_x)
    model_dist.load_weights(distortion_model)
    for i in tqdm(range(X_val1.shape[0])):
        dist_out = model_dist.predict(np.expand_dims(X_val1[i], 0))[0]
        dist_prob_mL.append(dist_out[0])

def cont_predict_model(X_val1, cont_prob_mL):
    print('Uploading Contrast model...')
    from keras.layers import Dense, Dropout
    from keras.applications import MobileNet
    from keras.models import Model
    cont_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    cont_x = Dropout(0.75)(cont_model.output)
    cont_x = Dense(2, activation='softmax')(cont_x)
    model_cont = Model(cont_model.input, cont_x)
    model_cont.load_weights(contrast_model)
    for i in tqdm(range(X_val1.shape[0])):
        cont_out = model_cont.predict(np.expand_dims(X_val1[i], 0))[0]
        cont_prob_mL.append(cont_out[0])

def white_predict_model(X_val1, white_prob_mL):
    print('Uploading White Balance model...')
    from keras.layers import Dense, Dropout
    from keras.applications import MobileNet
    from keras.models import Model
    whi_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    whi_x = Dropout(0.75)(whi_model.output)
    whi_x = Dense(2, activation='softmax')(whi_x)
    model_white = Model(whi_model.input, whi_x)
    model_white.load_weights(white_model)
    for i in tqdm(range(X_val1.shape[0])):
        white_out = model_white.predict(np.expand_dims(X_val1[i], 0))[0]
        white_prob_mL.append(white_out[0])

def sat_predict_model(X_val1, sat_prob_mL):
    print('Uploading Saturation model...')
    from keras.layers import Dense, Dropout
    from keras.applications import MobileNet
    from keras.models import Model
    sat_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    sat_x = Dropout(0.75)(sat_model.output)
    sat_x = Dense(2, activation='softmax')(sat_x)
    model_sat = Model(sat_model.input, sat_x)
    model_sat.load_weights(saturation_model)
    for i in tqdm(range(X_val1.shape[0])):
        sat_out = model_sat.predict(np.expand_dims(X_val1[i], 0))[0]
        sat_prob_mL.append(sat_out[0])

def sharp_predict_model(X_val1, sharp_prob_mL):
    print('Uploading Sharpened model...')
    from keras.layers import Dense, Dropout
    from keras.applications import MobileNet
    from keras.models import Model
    sharp_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    sharp_x = Dropout(0.75)(sharp_model.output)
    sharp_x = Dense(2, activation='softmax')(sharp_x)
    model_sharp = Model(sharp_model.input, sharp_x)
    model_sharp.load_weights(sharpness_model)
    for i in tqdm(range(X_val1.shape[0])):
        sharp_out = model_sharp.predict(np.expand_dims(X_val1[i], 0))[0]
        sharp_prob_mL.append(sharp_out[0])

def st_predict_model(X_val1, st_prob_mL):
    print('Uploading Straight line model...')
    from keras.layers import Dense, Dropout
    from keras.applications import MobileNet
    from keras.models import Model
    st_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    st_x = Dropout(0.75)(st_model.output)
    st_x = Dense(2, activation='softmax')(st_x)
    model_st = Model(st_model.input, st_x)
    model_st.load_weights(straight_model)
    for i in tqdm(range(X_val1.shape[0])):
        st_out = model_st.predict(np.expand_dims(X_val1[i], 0))[0]
        st_prob_mL.append(st_out[0])

def bright_predict_model(X_val1, bright_prob_mL):
    print('Uploading Brightness model...')
    from keras.layers import Dense, Dropout
    from keras.applications import MobileNet
    from keras.models import Model
    bright_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    bright_x = Dropout(0.75)(bright_model.output)
    bright_x = Dense(2, activation='softmax')(bright_x)
    model_bright = Model(bright_model.input, bright_x)
    model_bright.load_weights(brightness_model)
    for i in tqdm(range(X_val1.shape[0])):
        bright_out = model_bright.predict(np.expand_dims(X_val1[i], 0))[0]
        bright_prob_mL.append(bright_out[0])

def exp_predict_model(X_val1, exp_prob_mL):
    print('Uploading Exposure model...')
    from keras.layers import Dense, Dropout
    from keras.models import Model
    from keras import Sequential
    from keras.layers import Convolution2D as Conv2D
    from keras.layers import BatchNormalization
    from keras.layers import MaxPooling2D
    from keras.layers import Flatten
    exp_model = Sequential()

    exp_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    exp_model.add(BatchNormalization())
    exp_model.add(MaxPooling2D(pool_size=(2, 2)))

    exp_model.add(Conv2D(32, (3, 3), activation='relu'))
    exp_model.add(BatchNormalization())
    exp_model.add(MaxPooling2D(pool_size=(2, 2)))

    exp_model.add(Conv2D(64, (3, 3), activation='relu'))
    exp_model.add(BatchNormalization())
    exp_model.add(MaxPooling2D(pool_size=(2, 2)))

    exp_model.add(Conv2D(64, (3, 3), activation='relu'))
    exp_model.add(BatchNormalization())
    exp_model.add(MaxPooling2D(pool_size=(2, 2)))

    exp_model.add(Conv2D(128, (3, 3), activation='relu'))
    exp_model.add(BatchNormalization())
    exp_model.add(MaxPooling2D(pool_size=(2, 2)))

    exp_model.add(Conv2D(128, (3, 3), activation='relu'))
    exp_model.add(BatchNormalization())
    exp_model.add(MaxPooling2D(pool_size=(2, 2)))
    exp_model.add(Dropout(0.25))

    exp_model.add(Flatten())

    exp_model.add(Dense(512, activation='relu'))
    exp_model.add(BatchNormalization())
    exp_model.add(Dropout(0.5))
    exp_model.add(Dense(2, activation='softmax'))

    model_exp = Model(exp_model.input, exp_model.output)
    model_exp.load_weights(exposure_model)
    for i in tqdm(range(X_val1.shape[0])):
        exp_out = model_exp.predict(np.expand_dims(X_val1[i], 0))[0]
        exp_prob_mL.append(exp_out[0])