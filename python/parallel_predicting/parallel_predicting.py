from multiprocessing import Manager, Process
from python.parallel_predicting.predict_models import *

def predict_parallel(X_val1):
    m = Manager()
    aes_mean_mL = m.list()
    tech_mean_mL = m.list()
    angle_prob_mL = m.list()
    comp_prob_mL = m.list()
    dist_prob_mL = m.list()
    cont_prob_mL = m.list()
    white_prob_mL = m.list()
    sat_prob_mL = m.list()
    sharp_prob_mL = m.list()
    st_prob_mL = m.list()
    bright_prob_mL = m.list()
    exp_prob_mL = m.list()

    aes_mean_model = Process(target=aes_predict_model,args=[X_val1,aes_mean_mL])
    aes_mean_model.start()


    tech_mean_model = Process(target=tech_predict_model,args=[X_val1,tech_mean_mL])
    tech_mean_model.start()

    angle_mean_model = Process(target=angle_predict_model,args=[X_val1,angle_prob_mL])
    angle_mean_model.start()

    comp_mean_model = Process(target=comp_predict_model,args=[X_val1,comp_prob_mL])
    comp_mean_model.start()

    dist_mean_model = Process(target=dist_predict_model,args=[X_val1,dist_prob_mL])
    dist_mean_model.start()

    cont_mean_model = Process(target=cont_predict_model,args=[X_val1,cont_prob_mL])
    cont_mean_model.start()

    white_mean_model = Process(target=white_predict_model,args=[X_val1,white_prob_mL])
    white_mean_model.start()

    sat_mean_model = Process(target=sat_predict_model,args=[X_val1,sat_prob_mL])
    sat_mean_model.start()

    sharp_mean_model = Process(target=sharp_predict_model,args=[X_val1,sharp_prob_mL])
    sharp_mean_model.start()

    st_mean_model = Process(target=st_predict_model,args=[X_val1,st_prob_mL])
    st_mean_model.start()

    bright_mean_model = Process(target=bright_predict_model,args=[X_val1,bright_prob_mL])
    bright_mean_model.start()

    exp_mean_model = Process(target=exp_predict_model,args=[X_val1,exp_prob_mL])
    exp_mean_model.start()

    aes_mean_model.join()
    aes_mean = [x for x in aes_mean_mL]

    tech_mean_model.join()
    tech_mean = [x for x in tech_mean_mL]

    angle_mean_model.join()
    angle_prob = [x for x in angle_prob_mL]

    comp_mean_model.join()
    comp_prob = [x for x in comp_prob_mL]

    dist_mean_model.join()
    dist_prob = [x for x in dist_prob_mL]

    cont_mean_model.join()
    cont_prob = [x for x in cont_prob_mL]

    white_mean_model.join()
    white_prob = [x for x in white_prob_mL]

    sat_mean_model.join()
    sat_prob = [x for x in sat_prob_mL]

    sharp_mean_model.join()
    sharp_prob = [x for x in sharp_prob_mL]

    st_mean_model.join()
    st_prob = [x for x in st_prob_mL]

    bright_mean_model.join()
    bright_prob = [x for x in bright_prob_mL]

    exp_mean_model.join()
    exp_prob = [x for x in exp_prob_mL]

    return aes_mean, tech_mean, angle_prob, comp_prob, dist_prob, cont_prob, white_prob, sat_prob, sharp_prob, st_prob, bright_prob, exp_prob