from tqdm import tqdm
import numpy as np
import pandas as pd

def aes_scoring(X_val1, images1, aes_model, tech_model, model_angle, model_comp, model_dist, model_cont, model_white,
                model_sat, model_sharp, model_st, model_bright, model_exp, aes_grad, tech_grad):
    aes_mean = []
    tech_mean = []
    angle_prob = []
    comp_prob = []
    dist_prob = []
    cont_prob = []
    white_prob = []
    sat_prob = []
    sharp_prob = []
    st_prob = []
    bright_prob = []
    exp_prob = []

    for i in tqdm(range(X_val1.shape[0])):
        aes_out = aes_model.predict(np.expand_dims(X_val1[i], 0))[0]
        a_mean = 1 * aes_out[0] + 2 * aes_out[1] + 3 * aes_out[2] + 4 * aes_out[3] + 5 * aes_out[4] + 6 * \
                 aes_out[
                     5] + 7 * aes_out[6] + 8 * aes_out[7] + 9 * aes_out[8] + 10 * aes_out[9]
        aes_mean.append(a_mean)

        tech_out = tech_model.predict(np.expand_dims(X_val1[i], 0))[0]
        t_mean = 1 * tech_out[0] + 2 * tech_out[1] + 3 * tech_out[2] + 4 * tech_out[3] + 5 * tech_out[4] + 6 * \
                 tech_out[5] + 7 * tech_out[6] + 8 * tech_out[7] + 9 * tech_out[8] + 10 * tech_out[9]
        tech_mean.append(t_mean)

        ang_out = model_angle.predict(np.expand_dims(X_val1[i], 0))[0]
        angle_prob.append(ang_out[0])

        comp_out = model_comp.predict(np.expand_dims(X_val1[i], 0))[0]
        comp_prob.append(comp_out[0])

        dist_out = model_dist.predict(np.expand_dims(X_val1[i], 0))[0]
        dist_prob.append(dist_out[0])

        cont_out = model_cont.predict(np.expand_dims(X_val1[i], 0))[0]
        cont_prob.append(cont_out[0])

        white_out = model_white.predict(np.expand_dims(X_val1[i], 0))[0]
        white_prob.append(white_out[0])

        sat_out = model_sat.predict(np.expand_dims(X_val1[i], 0))[0]
        sat_prob.append(sat_out[0])

        sharp_out = model_sharp.predict(np.expand_dims(X_val1[i], 0))[0]
        sharp_prob.append(sharp_out[0])

        st_out = model_st.predict(np.expand_dims(X_val1[i], 0))[0]
        st_prob.append(st_out[0])

        bright_out = model_bright.predict(np.expand_dims(X_val1[i], 0))[0]
        bright_prob.append(bright_out[0])

        exp_out = model_exp.predict(np.expand_dims(X_val1[i], 0))[0]
        exp_prob.append(exp_out[0])

    aes_tech_data = pd.DataFrame()
    aes_tech_data['Image_Name'] = images1
    aes_tech_data['aes_mean'] = aes_mean
    aes_tech_data['tech_mean'] = tech_mean

    # aes_tech_data['aes_mean']= aes_tech_data['aes'].apply(ifef6_80)
    # aes_tech_data['tech_mean']= aes_tech_data['tech'].apply(tech)
    # aes_tech_data['aes_mean']= aes_tech_data['aes_mean'].apply(multiplier)
    # aes_tech_data['tech_mean']= aes_tech_data['tech_mean'].apply(multiplier)
    aes_tech_data['angle'] = angle_prob
    aes_tech_data['angle'] = np.where(aes_tech_data['angle'] > 0.4, 1, 0)
    aes_tech_data['comp'] = comp_prob
    aes_tech_data['comp'] = np.where(aes_tech_data['comp'] > 0.5, 1, 0)
    aes_tech_data['dist'] = dist_prob
    aes_tech_data['dist'] = np.where(aes_tech_data['dist'] > 0.5, 1, 0)
    aes_tech_data['cont'] = cont_prob
    aes_tech_data['white'] = white_prob
    aes_tech_data['sat'] = sat_prob
    aes_tech_data['sharp'] = sharp_prob
    aes_tech_data['st'] = st_prob
    aes_tech_data['st'] = np.where(aes_tech_data['st'] > 0.4, 1, 0)
    aes_tech_data['bright'] = bright_prob
    aes_tech_data['exp'] = exp_prob
    aes_tech_data['aes_mean_grad'] = aes_tech_data['aes_mean'].apply(aes_grad)
    # aes_tech_data['aes_mean_grad']= aes_tech_data['aes_mean_grad'].apply(multiplier)

    aes_tech_data['tech_mean_grad'] = aes_tech_data['aes_mean'].apply(tech_grad)
    # aes_tech_data['tech_mean_grad']= aes_tech_data['tech_mean_grad'].apply(multiplier)

    aes_score = sum(aes_tech_data['aes_mean']) / len(aes_tech_data['aes_mean'])
    tech_score = sum(aes_tech_data['tech_mean']) / len(aes_tech_data['tech_mean'])

    angle_score = sum(aes_tech_data['angle']) / len(aes_tech_data['angle']) * 10
    comp_score = sum(aes_tech_data['comp']) / len(aes_tech_data['comp']) * 10
    dist_score = sum(aes_tech_data['dist']) / len(aes_tech_data['dist']) * 10
    cont_score = sum(aes_tech_data['cont']) / len(aes_tech_data['cont']) * 10
    white_score = sum(aes_tech_data['white']) / len(aes_tech_data['white']) * 10
    sat_score = sum(aes_tech_data['sat']) / len(aes_tech_data['sat']) * 10
    sharp_score = sum(aes_tech_data['sharp']) / len(aes_tech_data['sharp']) * 10
    st_score = sum(aes_tech_data['st']) / len(aes_tech_data['st']) * 10
    bright_score = sum(aes_tech_data['bright']) / len(aes_tech_data['bright']) * 10
    exp_score = sum(aes_tech_data['exp']) / len(aes_tech_data['exp']) * 10
    # aes_score = sum(aes_tech_data['aes_mean_grad'])/len(aes_tech_data['aes_mean_grad'])
    # tech_score = sum(aes_tech_data['tech_mean_grad'])/len(aes_tech_data['tech_mean_grad'])
    # aes_score_mult=[i*1.5 if i*1.5<10 else 10 for i in aes_score]
    # tech_score_mult=[i*1.5 if i*1.5<10 else 10 for i in tech_score]

    return aes_tech_data, aes_score, tech_score, angle_score, comp_score, dist_score, cont_score, white_score, sat_score, sharp_score, st_score, bright_score, exp_score