from __future__ import print_function

import sys
import os
import re
import json
import requests
import warnings
import itertools
import PIL.ExifTags
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from os import listdir
from itertools import chain
from os.path import isfile, join
from PIL import ImageFile, ImageStat
from keras.preprocessing import image
from multiprocessing import Manager, Process
from datetime import datetime

ImageFile.LOAD_TRUNCATED_IMAGES = True

#print(keras.__version__)
#print(tf.__version__)
os.getcwd()

warnings.filterwarnings('ignore')

try:
    country = sys.argv[1]
    version = sys.argv[2]
except:
    pass
    #print("error in getting command line args")

cwd = os.getcwd()
#print("current working directory: " + cwd)

cwd = cwd + "/work/"

# Working directory
mypath = cwd + 'image_' + country + '/'
mypath2 = cwd

aesthetic_model = cwd + '../python/models/aesthetic_model_40_500_v27.hdf5'
technical_model = cwd + '../python/models/technical_model_30_500_v6.hdf5'

angle_model = cwd + '../python/models/mobilenet_Angle_50_52_4_64.hdf5'
white_model = cwd + '../python/models/mobilenet_White_50_70_4_64.hdf5'
contrast_model = cwd + '../python/models/mobilenet_Contrast_50_50_15_64.hdf5'
exposure_model = cwd + '../python/models/custom_model_Exposure_50_24_4_64.hdf5'
straight_model = cwd + '../python/models/mobilenet_Straight_50_60_4_64.hdf5'
sharpness_model = cwd + '../python/models/mobilenet_Sharpness_50_70_4_64.hdf5'
brightness_model = cwd + '../python/models/mobilenet_Brightness_50_50_5_64.hdf5'
distortion_model = cwd + '../python/models/mobilenet_Distortion_50_40_4_64.hdf5'
saturation_model = cwd + '../python/models/mobilenet_Saturation_50_70_4_64.hdf5'
composition_model = cwd + '../python/models/mobilenet_Composition_50_50_4_64.hdf5'

path = cwd


im_amenities_content = ['Banquet', 'Bar', 'Boardroom', 'Dining', 'Garden', 'Gym Spa', 'Restaurant', 'Parking',
                        'Gym Spa', 'Pool', 'Lawn', 'Lift', 'Kitchen']

im_amenities_content2 = ['Banquet Hall', 'Bar', 'Conference Room', 'Dining Area', 'Garden', 'Gym',
                         'In-house Restaurant', 'Parking Facility', 'Spa', 'Swimming Pool', 'Backyard', 'Elevator',
                         'Kitchen']

im_amenities_content_dict = {'Banquet Hall': 'Banquet',
                             'Bar': 'Bar',
                             'Conference Room': 'Boardroom',
                             'Dining Area': 'Dining',
                             'Garden': 'Garden',
                             'Gym': 'Gym Spa',
                             'In-house Restaurant': 'Restaurant',
                             'Parking Facility': 'Parking',
                             'Spa': 'Gym Spa',
                             'Swimming Pool': 'Pool',
                             'Backyard': 'Lawn',
                             'Elevator': 'Lift',
                             'Kitchen': 'Kitchen'}

def generate_tuple_string(string_list):
    tuple_string = str(tuple(string_list))
    if len(string_list) == 1:
        index = len(tuple_string) - 2  # removing trailing comma before closing parantheses
        tuple_string = tuple_string[:index] + tuple_string[index + 1:]

    return tuple_string


def new_category(category):
    cat = str(category).upper()
    cat = re.sub(" +", " ", re.sub("[^A-Za-z]+", " ", str(cat)))
    cat = cat.strip()

    if 'FAC' in cat:
        return 'FACADE'
    elif 'LOB' in cat:
        return 'LOBBY'
    elif ('WAS' in cat or 'WSH' in cat or 'BATH' in cat):
        return 'WASHROOM'
    elif (
            'MEETING' in cat or 'DRAWING' in cat or 'LIVING' in cat or 'DINING' in cat or 'CONFERENCE' in cat or 'GAME' in cat or 'PLAY' in cat or 'BOARD' in cat):
        return 'OTHERS'
    elif 'RECE' in cat:
        return 'RECEPTION'
    elif 'ROO' in cat:
        return 'ROOM'
    else:
        return 'OTHERS'


def get_oyo_ids():
    return ['MUM803']


def get_hotel_data():
    hotels = pd.DataFrame()
    hotels['oyo_id'] = get_oyo_ids()
    return hotels


def generate_and_get_property_mrc(oyo_ids, im_indonesia1):
    hotel_mrc = pd.read_csv(path + country + "_hotel_mrc.csv")
    hotel_mrc2 = hotel_mrc.groupby(['oyo_id'], as_index=False) \
        .agg({'mrc': 'count'}) \
        .rename(columns={'oyo_id': 'oyo_id', 'mrc': 'mrc_count'}) \
        .fillna(0)

    im_indonesia2 = im_indonesia1[im_indonesia1["new_cat"] == "ROOM"]
    im_indonesia2 = im_indonesia2.fillna(0)
    im_indonesia3 = im_indonesia2.groupby(['oyo_id', 'mrc_id'], as_index=False) \
        .agg({'hotel_image': 'count'}) \
        .rename(columns={'oyo_id': 'oyo_id', 'mrc_id': 'mrc_id', 'hotel_image': 'Image_Count'}) \
        .fillna(0)
    im_indonesia3['Flag4'] = np.where(im_indonesia3['Image_Count'] >= 4, 1, 0)
    im_indonesia4 = im_indonesia3.groupby(['oyo_id'], as_index=False) \
        .agg({'Flag4': 'sum'}) \
        .rename(columns={'oyo_id': 'oyo_id', 'Flag4': 'Flag4_sum'}) \
        .fillna(0)
    im_indonesia4 = pd.merge(im_indonesia4, hotel_mrc2, on=['oyo_id'], how='left')
    im_indonesia4['mrc_count'] = im_indonesia4['mrc_count'].fillna(0)
    im_indonesia4['room_quant4'] = (
        np.where(im_indonesia4['mrc_count'] == 0, 0, round(im_indonesia4['Flag4_sum'] / im_indonesia4['mrc_count'], 2)))
    im_indonesia4['room_quant4'] = np.where(im_indonesia4['room_quant4'] > 1, 1, im_indonesia4['room_quant4'])
    im_indonesia5 = im_indonesia4[['oyo_id', 'mrc_count', 'room_quant4']]
    im_indonesia5.to_csv(path + country + "_room_quant_score.csv", index=False)
    #print("generated MRC successfully")
    return im_indonesia5


def download_images(url):
    try:
        #print("downloading " + url)
        #print(path)
        filename = url.split('/')[-1]
        r = requests.get(url, allow_redirects=True)
        open(path + "image_" + country + "/" + "_" + filename + ".jpg",
             'wb').write(r.content)
    except:
        pass
        #print("error while downloading")


def get_amenities_as_list():
    hotel_amenities = pd.read_csv(path + country + '_amenities.csv', encoding='latin-1')
    hotel_amenities['hotel_amenities'] = hotel_amenities['hotel_amenities'].fillna('no Amenities')
    hotel_am = hotel_amenities.set_index('oyo_id').T.to_dict('list')
    return hotel_am


def get_room_quant_score():
    hotel_amenities = pd.read_csv(path + country + '_amenities.csv', encoding='latin-1')
    hotel_amenities['hotel_amenities'] = hotel_amenities['hotel_amenities'].fillna('no Amenities')
    room_quant_score = pd.read_csv(path + country + '_room_quant_score.csv', encoding='latin-1')
    room_quant_sc = room_quant_score.set_index('oyo_id').T.to_dict('list')
    return room_quant_sc


def aes_grad(col):
    if col <= 1.00007:
        return 0 + ((col - 0) / (1.00007 - 0))
    if col > 1.00007 and col <= 1.32409:
        return 1 + ((col - 1.00007) / (1.00007 - 7.236282))
    if col > 1.32409 and col <= 2.95663:
        return 2 + ((col - 1.32409) / (2.95663 - 1.32409))
    if col > 2.95663 and col <= 4.76146:
        return 3 + ((col - 2.95663) / (4.76146 - 2.95663))
    if col > 4.76146 and col <= 6.59607:
        return 4 + ((col - 4.76146) / (6.59607 - 4.76146))
    if col > 5.99812 and col <= 6.59607:
        return 5 + ((col - 5.99812) / (6.59607 - 5.99812))
    if col > 6.59607 and col <= 7.10172:
        return 6 + ((col - 6.59607) / (7.10172 - 6.59607))
    if col > 7.10172 and col <= 7.82517:
        return 7 + ((col - 7.10172) / (7.82517 - 7.10172))
    if col > 7.82517 and col <= 8.00960:
        return 8 + ((col - 7.82517) / (8.00960 - 7.82517))
    if col > 8.00960:
        return 9 + ((col - 8.00960) / (10 - 8.00960))


# Gradient_tech_new_model
def tech_grad(col):
    if col <= 1.15888:
        return 0 + ((col - 0) / (1.15888 - 0))
    if col > 1.15888 and col <= 2.22530:
        return 1 + ((col - 1.15888) / (2.22530 - 1.15888))
    if col > 2.22530 and col <= 3.56137:
        return 2 + ((col - 2.22530) / (3.56137 - 2.22530))
    if col > 3.56137 and col <= 4.82128:
        return 3 + ((col - 3.56137) / (4.82128 - 3.56137))
    if col > 4.82128 and col <= 5.13893:
        return 4 + ((col - 4.82128) / (5.13893 - 4.82128))
    if col > 5.13893 and col <= 5.99385:
        return 5 + ((col - 5.13893) / (5.99385 - 5.13893))
    if col > 5.99385 and col <= 6.62869:
        return 6 + ((col - 5.99385) / (6.62869 - 5.99385))
    if col > 6.62869 and col <= 7.36670:
        return 7 + ((col - 6.62869) / (7.36670 - 6.62869))
    if col > 7.36670 and col <= 8.86024:
        return 8 + ((col - 7.36670) / (8.86024 - 7.36670))
    if col > 8.86024:
        return 9 + ((col - 8.86024) / (10 - 8.86024))


def multiplier(col):
    col = col * 1.5
    if col > 10:
        return 10
    else:
        return col


# FUZZY IMAGE SIMILARITY MATCHING ALGORITHM
def conv_to_cat(n):
    if n < 0.5:
        return 'a'
    elif 0.05 <= n < 0.15:
        return 'b'
    elif 0.15 <= n < 0.25:
        return 'c'
    elif 0.25 <= n < 0.35:
        return 'd'
    elif 0.35 <= n < 0.45:
        return 'e'
    elif 0.45 <= n < 0.55:
        return 'f'
    elif 0.55 <= n < 0.65:
        return 'g'
    elif 0.65 <= n < 0.75:
        return 'h'
    elif 0.75 <= n < 0.85:
        return 'i'
    elif 0.85 <= n < 0.95:
        return 'j'
    elif 0.95 <= n < 1:
        return 'k'
    else:
        return 'x'


# Token Based
# Jaccard
def qgram(str1, str2, q=2, common_divisor='longest', min_threshold=None, padded=False):
    if (str1 == '') or (str2 == '') or isinstance(str1, float) or isinstance(str2, float):
        return 0.0
    elif (str1 == str2):
        return 1.0

    # Calculate number of q-grams in strings (plus start and end characters) - -
    if (padded == True):
        num_qgram1 = len(str1) + q - 1
        num_qgram2 = len(str2) + q - 1
    else:
        num_qgram1 = max(len(str1) - (q - 1), 0)  # Make sure its not negative
        num_qgram2 = max(len(str2) - (q - 1), 0)

    # Check if there are q-grams at all from both strings - - - - - - - - - - - -
    # (no q-grams if length of a string is less than q)
    if ((padded == False) and (min(num_qgram1, num_qgram2) == 0)):
        return 0.0

    # Calculate the divisor - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if (common_divisor == 'average'):
        divisor = 0.5 * (num_qgram1 + num_qgram2)  # Compute average number of q-grams
    elif (common_divisor == 'shortest'):
        divisor = min(num_qgram1, num_qgram2)
    else:  # Longest
        divisor = max(num_qgram1, num_qgram2)

    # Use number of q-grams to quickly check for minimum threshold - - - - - - -
    if (min_threshold != None):
        if (isinstance(min_threshold, float)) and (min_threshold > 0.0) and \
                (min_threshold > 0.0):
            max_common_qgram = min(num_qgram1, num_qgram2)

            w = float(max_common_qgram) / float(divisor)

        if (w < min_threshold):
            return 0.0  # Similariy is smaller than minimum threshold

    # Add start and end characters (padding) - - - - - - - - - - - - - - - - - -
    if (padded == True):
        qgram_str1 = (q - 1) * QGRAM_START_CHAR + str1 + (q - 1) * QGRAM_END_CHAR
        qgram_str2 = (q - 1) * QGRAM_START_CHAR + str2 + (q - 1) * QGRAM_END_CHAR
    else:
        qgram_str1 = str1
        qgram_str2 = str2

    # Make a list of q-grams for both strings - - - - - - - - - - - - - - - - - -
    qgram_list1 = [qgram_str1[i:i + q] for i in range(len(qgram_str1) - (q - 1))]
    qgram_list2 = [qgram_str2[i:i + q] for i in range(len(qgram_str2) - (q - 1))]

    # Get common q-grams  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    common = 0

    if (num_qgram1 < num_qgram2):  # Count using the shorter q-gram list
        short_qgram_list = qgram_list1
        long_qgram_list = qgram_list2
    else:
        short_qgram_list = qgram_list2
        long_qgram_list = qgram_list1

    for q_gram in short_qgram_list:
        if (q_gram in long_qgram_list):
            common += 1
            long_qgram_list.remove(q_gram)  # Remove the counted q-gram

    w = float(common) / float(divisor)

    assert (w >= 0.0) and (w <= 1.0), 'Similarity weight outside 0-1: %f' % (w)

    return w


# convert pixels to string
def img_to_str(image_path, image_name, size):
    img = image.load_img(image_path + image_name, target_size=(size, size, 3))
    img = image.img_to_array(img)
    img = img / 255
    img_to_arr = np.average(img, weights=[0.299, 0.587, 0.114], axis=2)
    vec = np.vectorize(conv_to_cat)
    cat_arr = vec(img_to_arr)
    row_arr = [''.join(cat_arr[_, :]) for _ in range(img_to_arr.shape[0])]
    col_arr = [''.join(cat_arr[:, _]) for _ in range(img_to_arr.shape[1])]
    return row_arr + col_arr


# create similarity index
def similarity_index(list1, list2, q):
    if len(list1) == len(list2):
        jaccard_dist = [qgram(list1[i], list2[i], q) for i in range(len(list1))]
    else:
        jaccard_dist = []

    if len(jaccard_dist) > 0:
        return round(sum(jaccard_dist) / len(jaccard_dist), 4)
    else:
        return 0


# Helper functions
def filter_images(image_path, images):
    image_list = []
    for image in images:
        try:
            assert imread(image_path + image).shape[2] == 3
            image_list.append(image)
        except:
            pass
    return image_list


# Function to create image dictionary
def img_arr_dict(image_path, image_list, size):
    ds_dict = {}
    duplicates = []
    for image in image_list:
        match_arr = img_to_str(image_path, image, size)

        if image not in ds_dict:
            ds_dict[image] = match_arr
        else:
            duplicates.append((image, ds_dict[image], 1))
    return duplicates, ds_dict


# Similarity buckets
def similar_set(duplicates):
    #print('similar_sets start')
    if len(duplicates) > 0:
        dup1 = []
        dup2 = []
        for i in range(len(duplicates)):
            dup1.append((duplicates[i][0], duplicates[i][1]))
            dup1.append((duplicates[i][1], duplicates[i][0]))
            dup2.append(duplicates[i][0])
            dup2.append(duplicates[i][1])
        dup_unique = list(set(dup2))
        set1 = {}
        for i in range(len(dup_unique)):
            set1[i] = [dup_unique[i]]
            for j in range(len(dup1)):
                if dup_unique[i] == dup1[j][0]:
                    set1[i].append(dup1[j][1])
        sim_set = list(set([tuple(set(val)) for val in set1.values()]))
        #print('similar_set end')
    else:
        sim_set = []
    return sim_set


def image_color_detection(pil_img, thumb_size=40, MSE_cutoff=0, adjust_color_bias=True):
    bands = pil_img.getbands()
    if bands == ('R', 'G', 'B') or bands == ('R', 'G', 'B', 'A'):
        thumb = pil_img.resize((thumb_size, thumb_size))
        SSE, bias = 0, [0, 0, 0]
        if adjust_color_bias:
            bias = ImageStat.Stat(thumb).mean[:3]
            bias = [b - sum(bias) / 3 for b in bias]
        for pixel in thumb.getdata():
            mu = sum(pixel) / 3
            SSE += sum((pixel[i] - mu - bias[i]) * (pixel[i] - mu - bias[i]) for i in [0, 1, 2])
        MSE = float(SSE) / (thumb_size * thumb_size)
        if MSE <= MSE_cutoff:
            return 1
        else:
            return 0
    #        #print("( MSE=",MSE,")")
    elif len(bands) == 1:
        return 1
    else:
        return 0


def landscape(pil_img):
    if (pil_img.size[0]) >= (pil_img.size[1]):
        return 0
    else:
        return 1


def get_images_of_property(all_images, hotel_id):
    images = []
    for i in range(len(all_images)):
        if '_'.join((all_images[i].split('_')[0:2])) == hotel_id:
            images.append(all_images[i])
        elif '_'.join((all_images[i].split('_')[0:1])) == hotel_id:
            images.append(all_images[i])
    return images


def get_val_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224, 3))
    img = image.img_to_array(img)
    img = img / 255
    return img


def group_similar_images(images1):
    image_path = mypath
    image_files = filter_images(image_path, images1)
    duplicates, ds_dict = img_arr_dict(image_path, image_files, 25)
    for k1, k2 in tqdm(itertools.combinations(ds_dict, 2)):
        s_index = similarity_index(ds_dict[k1], ds_dict[k2], 2)
        if s_index == 1:
            duplicates.append((k1, k2, s_index))

    groups = similar_set(duplicates)
    return groups


def aes_scoring(X_val1, images1):
    aes_mean, tech_mean, angle_prob, comp_prob, dist_prob, cont_prob, white_prob, sat_prob, sharp_prob, st_prob, bright_prob, exp_prob = predict_parallel(
        X_val1)

    aes_tech_data = pd.DataFrame()
    aes_tech_data['Image_Name'] = images1
    aes_tech_data['aes_mean'] = aes_mean
    aes_tech_data['tech_mean'] = tech_mean

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

    return aes_tech_data, aes_score, tech_score, angle_score, comp_score, dist_score, cont_score, white_score, sat_score, sharp_score, st_score, bright_score, exp_score


def append_aes_scoring_comments(model_comments, aes_tech_data, aes_score, tech_score, angle_score, comp_score,
                                dist_score, cont_score, white_score, sat_score, sharp_score, st_score, bright_score,
                                exp_score):
    if aes_score < 5:
        comment = 'Image overall aesthetics are not upto the mark for ' + str(int(
            (aes_tech_data[aes_tech_data.aes_mean < 5].count()['Image_Name']) / len(
                aes_tech_data['Image_Name']) * 100)) + '% of images'
        model_comments.append(comment)
    if tech_score < 5:
        comment = 'Image overall technicals are not upto the mark for ' + str(int(
            (aes_tech_data[aes_tech_data.tech_mean < 5].count()['Image_Name']) / len(
                aes_tech_data['Image_Name']) * 100)) + '% of images'
        model_comments.append(comment)
    if angle_score < 5:
        comment = 'Image angle positioning is not upto the mark for ' + str(int(
            (aes_tech_data[aes_tech_data.angle == 0].count()['Image_Name']) / len(
                aes_tech_data['Image_Name']) * 100)) + '% of images'
        model_comments.append(comment)
    if comp_score < 5:
        comment = 'Image composition is not upto the mark for ' + str(int(
            (aes_tech_data[aes_tech_data.comp == 0].count()['Image_Name']) / len(
                aes_tech_data['Image_Name']) * 100)) + '% of images'
        model_comments.append(comment)
    if dist_score < 5:
        comment = 'Image blurrnes and barrel distortion is present for ' + str(int(
            (aes_tech_data[aes_tech_data.dist == 0].count()['Image_Name']) / len(
                aes_tech_data['Image_Name']) * 100)) + '% of images'
        model_comments.append(comment)
    if cont_score < 5:
        comment = 'Image contrast is not upto the mark for ' + str(int(
            (aes_tech_data[aes_tech_data.cont < 0.5].count()['Image_Name']) / len(
                aes_tech_data['Image_Name']) * 100)) + '% of images'
        model_comments.append(comment)
    if white_score < 5:
        comment = 'Image white balance is not upto the mark for ' + str(int(
            (aes_tech_data[aes_tech_data.white < 0.5].count()['Image_Name']) / len(
                aes_tech_data['Image_Name']) * 100)) + '% of images'
        model_comments.append(comment)
    if sat_score < 5:
        comment = 'Image saturation is not upto the mark for ' + str(int(
            (aes_tech_data[aes_tech_data.sat < 0.5].count()['Image_Name']) / len(
                aes_tech_data['Image_Name']) * 100)) + '% of images'
        model_comments.append(comment)
    if sharp_score < 5:
        comment = 'Image sharpness is not upto the mark for ' + str(int(
            (aes_tech_data[aes_tech_data.sharp < 0.5].count()['Image_Name']) / len(
                aes_tech_data['Image_Name']) * 100)) + '% of images'
        model_comments.append(comment)
    if st_score < 5:
        comment = 'Image alignment is not upto the mark for ' + str(int(
            (aes_tech_data[aes_tech_data.st == 0].count()['Image_Name']) / len(
                aes_tech_data['Image_Name']) * 100)) + '% of images'
        model_comments.append(comment)
    if bright_score < 5:
        comment = 'Image brightness is not upto the mark for ' + str(int(
            (aes_tech_data[aes_tech_data.st < 0.5].count()['Image_Name']) / len(
                aes_tech_data['Image_Name']) * 100)) + '% of images'
        model_comments.append(comment)
    if exp_score < 5:
        comment = 'Image exposure is not upto the mark for ' + str(int(
            (aes_tech_data[aes_tech_data.st < 0.5].count()['Image_Name']) / len(
                aes_tech_data['Image_Name']) * 100)) + '% of images'
        model_comments.append(comment)

    return model_comments


def exif_scoring(images1):
    # Technical Exif scoring logic
    exif_score = []
    null_count = 0
    flag_dim_v1 = 0
    flag_dim_v2 = 0
    flag_iso_v1 = 0
    flag_iso_v2 = 0
    flag_res_v1 = 0
    flag_res_v2 = 0
    flag_ape_v1 = 0
    flag_ape_v2 = 0
    bwhite = 0
    lscape = 0

    start_time = datetime.now()

    for j in range(len(images1)):
        img = Image.open(mypath + images1[j])
        try:
            bwhite += image_color_detection(img, 40, 22, True)
        except:
            pass

        try:
            lscape += landscape(img)
        except:
            pass

        try:
            if img._getexif():
                exif = {
                    PIL.ExifTags.TAGS[k]: v
                    for k, v in img._getexif().items()
                    if k in PIL.ExifTags.TAGS
                }
            else:
                exif = {}
        except:
            from PIL.TiffTags import TAGS

            exif = {TAGS[key]: img.tag[key] for key in img.tag.keys()}

        ex_score = 0
        if bool(exif) == False:
            null_count += 1
        else:
            if 'ExifImageHeight' in exif and 'ExifImageWidth' in exif:
                if exif["ExifImageHeight"] == 1920 and exif["ExifImageWidth"] == 2880:
                    ex_score += 2.5
                else:
                    flag_dim_v1 += 1  # Image Dimension creteria is not satisfied
            else:
                flag_dim_v2 += 1  # EXIF details are not present for Dimensions

            if 'ISOSpeedRatings' in exif:
                if exif["ISOSpeedRatings"] <= 200:
                    ex_score += 2.5 * 1.0
                elif 200 < exif["ISOSpeedRatings"] <= 400:
                    ex_score += 2.5 * 0.7
                elif 400 < exif["ISOSpeedRatings"] <= 800:
                    ex_score += 2.5 * 0.4
                else:
                    flag_iso_v1 += 1  # Image ISO creteria is not satisfied
            else:
                flag_iso_v2 += 1  # EXIF details are not present for ISO

            try:
                if 'XResolution' in exif:
                    if exif["XResolution"][0] / exif["XResolution"][1] >= 300:
                        ex_score += 2.5 * 1.0
                    elif 200 <= exif["XResolution"][0] / exif["XResolution"][1] < 300:
                        ex_score += 2.5 * 0.7
                    elif 150 <= exif["XResolution"][0] / exif["XResolution"][1] < 200:
                        ex_score += 2.5 * 0.4
                    else:
                        flag_res_v1 += 1  # Image Resolution creteria is not satisfied
                else:
                    flag_res_v2 += 1  # EXIF details are not present for Resolution
            except:
                if 'XResolution' in exif:
                    if exif["XResolution"][0][0] / exif["XResolution"][0][1] >= 300:
                        ex_score += 2.5 * 1.0
                    elif 200 <= exif["XResolution"][0][0] / exif["XResolution"][0][1] < 300:
                        ex_score += 2.5 * 0.7
                    elif 150 <= exif["XResolution"][0][0] / exif["XResolution"][0][1] < 200:
                        ex_score += 2.5 * 0.4
                    else:
                        flag_res_v1 += 1  # Image Resolution creteria is not satisfied
                else:
                    flag_res_v2 += 1  # EXIF details are not present for Resolution

            if 'FNumber' in exif:
                if 4 <= exif["FNumber"][0] / exif["FNumber"][1] < 8:
                    ex_score += 2.5 * 1.0
                elif 8 <= exif["FNumber"][0] / exif["FNumber"][1] < 12:
                    ex_score += 2.5 * 0.7
                elif 12 <= exif["FNumber"][0] / exif["FNumber"][1] < 14:
                    ex_score += 2.5 * 0.4
                else:
                    flag_ape_v1 += 1  # Image Aperture creteria is not satisfied
            else:
                flag_ape_v2 += 1  # EXIF details are not present for Apreture

        exif_score.append(ex_score)
        if len(exif_score) > 0:
            tech_exif_score = round(sum(exif_score) / len(exif_score), 4)
        else:
            tech_exif_score = 0

        end_time = datetime.now()
        #print("time taken for exif scoring: " + str(end_time - start_time))
        return exif_score, tech_exif_score, null_count, flag_dim_v1, flag_dim_v2, flag_iso_v1, flag_iso_v2, flag_res_v1, flag_res_v2, flag_ape_v1, flag_ape_v2, bwhite, lscape


def append_exif_scoring_comments(model_comments, images1, tech_exif_score, null_count, flag_dim_v1, flag_dim_v2,
                                 flag_iso_v1, flag_iso_v2, flag_res_v1, flag_res_v2, flag_ape_v1, flag_ape_v2, bwhite,
                                 lscape):
    if bwhite > 0:
        bwhite_score = 0
        comment = str(bwhite) + ' images are Black and White for this property'
        model_comments.append(comment)
    else:
        bwhite_score = 10
    if lscape > 0:
        lscape_score = 0
        comment = str(lscape) + ' images are not in landscape mode for this property'
        model_comments.append(comment)
    else:
        lscape_score = 10
    if null_count / len(images1) > 0:
        comment = 'Overall EXIF details are not present for ' + str(
            round(null_count / len(images1) * 100, 1)) + '% of images'
    #        model_comments.append(comment)
    if flag_dim_v1 / len(images1) > 0:
        comment = 'Image Dimension creteria is not satisfied for ' + str(
            round(flag_dim_v1 / len(images1) * 100, 1)) + '% of images'
    #        model_comments.append(comment)
    if flag_dim_v2 / len(images1) > 0:
        comment = 'EXIF details are not present for Dimension in ' + str(
            round(flag_dim_v2 / len(images1) * 100, 1)) + '% of images'
    #        model_comments.append(comment)
    if flag_iso_v1 / len(images1) > 0:
        comment = 'Image ISO creteria is not satisfied for ' + str(
            round(flag_iso_v1 / len(images1) * 100, 1)) + '% of images'
    #        model_comments.append(comment)
    if flag_iso_v2 / len(images1) > 0:
        comment = 'EXIF details are not present for ISO in ' + str(
            round(flag_iso_v2 / len(images1) * 100, 1)) + '% of images'
    #        model_comments.append(comment)
    if flag_res_v1 / len(images1) > 0:
        comment = 'Image Resolution creteria is not satisfied for ' + str(
            round(flag_res_v1 / len(images1) * 100, 1)) + '% of images'
    #        model_comments.append(comment)
    if flag_res_v2 / len(images1) > 0:
        comment = 'EXIF details are not present for Resolution in ' + str(
            round(flag_res_v2 / len(images1) * 100, 1)) + '% of images'
    #        model_comments.append(comment)
    if flag_ape_v1 / len(images1) > 0:
        comment = 'Image Aperture creteria is not satisfied for ' + str(
            round(flag_ape_v1 / len(images1) * 100, 1)) + '% of images'
    #        model_comments.append(comment)
    if flag_ape_v2 / len(images1) > 0:
        comment = 'EXIF details are not present for Aperture in ' + str(
            round(flag_ape_v2 / len(images1) * 100, 1)) + '% of images'
    #        model_comments.append(comment)

    return model_comments, bwhite_score, lscape_score


def aes_predict_model(X_val1, aes_mean_mL):
    #print('Uploading Aesthetic model...')
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
    #print('Uploading Technical model...')
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
    #print('Uploading Angle positioning model...')
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
    #print('Uploading Composition model...')
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
    #print('Uploading Distortion model...')
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
    #print('Uploading Contrast model...')
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
    #print('Uploading White Balance model...')
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
    #print('Uploading Saturation model...')
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
    #print('Uploading Sharpened model...')
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
    #print('Uploading Straight line model...')
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
    #print('Uploading Brightness model...')
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
    #print('Uploading Exposure model...')
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

    start_time = datetime.now()

    aes_mean_model = Process(target=aes_predict_model, args=[X_val1, aes_mean_mL])
    aes_mean_model.start()

    tech_mean_model = Process(target=tech_predict_model, args=[X_val1, tech_mean_mL])
    tech_mean_model.start()

    angle_mean_model = Process(target=angle_predict_model, args=[X_val1, angle_prob_mL])
    angle_mean_model.start()

    comp_mean_model = Process(target=comp_predict_model, args=[X_val1, comp_prob_mL])
    comp_mean_model.start()

    dist_mean_model = Process(target=dist_predict_model, args=[X_val1, dist_prob_mL])
    dist_mean_model.start()

    cont_mean_model = Process(target=cont_predict_model, args=[X_val1, cont_prob_mL])
    cont_mean_model.start()

    white_mean_model = Process(target=white_predict_model, args=[X_val1, white_prob_mL])
    white_mean_model.start()

    sat_mean_model = Process(target=sat_predict_model, args=[X_val1, sat_prob_mL])
    sat_mean_model.start()

    sharp_mean_model = Process(target=sharp_predict_model, args=[X_val1, sharp_prob_mL])
    sharp_mean_model.start()

    st_mean_model = Process(target=st_predict_model, args=[X_val1, st_prob_mL])
    st_mean_model.start()

    bright_mean_model = Process(target=bright_predict_model, args=[X_val1, bright_prob_mL])
    bright_mean_model.start()

    exp_mean_model = Process(target=exp_predict_model, args=[X_val1, exp_prob_mL])
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

    end_time = datetime.now()
    #print("time taken to upload models and predict: " + str(end_time - start_time))

    return aes_mean, tech_mean, angle_prob, comp_prob, dist_prob, cont_prob, white_prob, sat_prob, sharp_prob, st_prob, bright_prob, exp_prob


def main():
    # hotels_df = get_hotel_data()
    # im_indonesia1 = pd.read_csv(path + country + "_images_link.csv")
    # im_amenities = pd.read_csv(path + country + "_amenities.csv")
    # im_indonesia5 = generate_and_get_property_mrc(hotels_df['oyo_id'], im_indonesia1)
    download_images(sys.argv[3])
    # hotel_am = get_amenities_as_list()
    # room_quant_sc = get_room_quant_score()

    # img_link = pd.read_csv(path + country + '_images_link.csv', encoding='latin-1')
    # properties = [f for f in img_link['oyo_id'].unique()]
    # img_link = "https://images.oyoroomscdn.com/uploads/hotel_image/82669/aab9037a45518738.jpg"
    all_images = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and (
            f.endswith(".jpg.jpg") or f.endswith(".JPG.jpg") or f.endswith(".JPEG.jpg") or f.endswith(
        ".jpeg.jpg") or f.endswith(".PNG.jpg") or f.endswith(".png.jpg")))]

    final = []
    img_score_overall = pd.DataFrame()

    images = all_images

    image_path = mypath
    images1 = []
    val_image = []
    img = get_val_image(mypath + images[0])
    images1.append(images[0])
    val_image.append(img)

    X_val = np.array(val_image)

    groups = group_similar_images(images1)

    if len(groups) > 0:
        dis_sim = list(chain(*groups))
        dis_sim = set(dis_sim)
        dis_sim = list(dis_sim)
        sim_val_image = []
        for i in tqdm(range(len(dis_sim))):
            try:
                sim_img = image.load_img(image_path + '/' + dis_sim[i], target_size=(224, 224, 3))
                sim_img = image.img_to_array(sim_img)
                sim_img = sim_img / 255
                # images.append(dis_sim[i])
                sim_val_image.append(sim_img)
            except:
                continue

        sim_X_val = np.array(sim_val_image)
        m = Manager()
        aes_mean_ml = m.list()
        aes_mean_model = Process(target=aes_predict_model, args=[sim_X_val, aes_mean_ml])
        aes_mean_model.start()
        aes_mean_model.join()
        aes_mean = [x for x in aes_mean_ml]
        sim_aes = pd.DataFrame({'Image': dis_sim, 'Score': aes_mean})
        groups_copy = list(groups)

        for m in range(len(groups_copy)):
            groups_copy[m] = list(groups_copy[m])
        for m in range(len(groups_copy)):
            for i in range(len(groups_copy[m])):
                for j in range(len(sim_aes)):
                    if groups_copy[m][i] == sim_aes['Image'][j]:
                        groups_copy[m][i] = sim_aes['Score'][j]

        max_sim = []
        for i in groups_copy:
            x = max(i)
            max_sim.append(x)

        for i in range(len(groups_copy)):
            groups_copy[i].remove(max_sim[i])
        try:
            del to_remove
        except:
            pass

        remv = []
        for i in groups_copy:
            for j in i:
                remv.append(j)
        to_remove = pd.DataFrame(remv, columns=['Score'])
        to_remove = pd.merge(to_remove, sim_aes, on='Score', how='inner')
        to_remove.drop_duplicates(subset='Score', inplace=True)
        to_remove.reset_index(drop=True)
        to_remove = to_remove['Image'].to_list()
        images1 = list(set(images1) - set(to_remove))
        images = list(set(images) - set(to_remove))
    else:
        pass

    if len(images1) > 0:
        val_image = []
        try:
            img = get_val_image(image_path + '/' + images1[0])
            val_image.append(img)
        except:
            pass
            #print('image not uploaded')
        X_val1 = np.array(val_image)

    #print('test 1 completed: SIMILARITY SCREENING')

    model_comments = []

    aes_tech_data, aes_score, tech_score, angle_score, comp_score, dist_score, cont_score, white_score, sat_score, sharp_score, st_score, bright_score, exp_score = aes_scoring(
        X_val1, images1)
    #print('test 2 completed: AES SCORING')
    model_comments = append_aes_scoring_comments(model_comments, aes_tech_data, aes_score, tech_score,
                                                 angle_score, comp_score, dist_score, cont_score, white_score,
                                                 sat_score, sharp_score, st_score, bright_score, exp_score)

    #print('checkpoint 1 completed')

    exif_score, tech_exif_score, null_count, flag_dim_v1, flag_dim_v2, flag_iso_v1, flag_iso_v2, flag_res_v1, flag_res_v2, flag_ape_v1, flag_ape_v2, bwhite, lscape = exif_scoring(
        images1)
    #print('test 3 completed: EXIF SCORING')
    model_comments, bwhite_score, lscape_score = append_exif_scoring_comments(model_comments, images1,
                                                                              tech_exif_score, null_count,
                                                                              flag_dim_v1, flag_dim_v2,
                                                                              flag_iso_v1, flag_iso_v2,
                                                                              flag_res_v1, flag_res_v2,
                                                                              flag_ape_v1, flag_ape_v2, bwhite,
                                                                              lscape)
    #print('checkpoint 2 completed')

    # Quantity scoring logic
    category = []
    room_ct = 0
    facade_ct = 0
    wash_ct = 0
    recep_ct = 0
    lobby_ct = 0
    misc_ct = 0
    amen_list = []
    j = 0
    try:
        images1[0] = re.sub('_+', '_', images1[0])
        if '_'.join((images1[j].split('_')[0:2])):
            cat = images1[j].split('_')[2].strip()
        elif '_'.join((images1[i].split('_')[0:1])):
            cat = images1[j].split('_')[1].strip()
        if cat[0:4].upper() == 'ROOM':
            room_ct += 1
        elif cat[0:6].upper() == 'FACADE':
            facade_ct += 1
        elif cat[0:8].upper() == 'WASHROOM':
            wash_ct += 1
        elif cat[0:9].upper() == 'RECEPTION':
            recep_ct += 1
        elif cat[0:5].upper() == 'LOBBY':
            lobby_ct += 1
        else:
            misc_ct += 1
        category.append(cat)
    except:
        pass

    try:
        if '_'.join((images1[j].split('_')[0:2])):
            amen = images1[j].split('_')[3].strip()
            amen_list.append(amen)
        elif '_'.join((images1[i].split('_')[0:1])):
            amen = images1[j].split('_')[2].strip()
            amen_list.append(amen)
    except:
        pass
    # amen_list = list(set(amen_list))
    # hotel_am_list = hotel_am[hotel_id][0].split(',')
    # list1 = list(set(hotel_am_list).intersection(set(im_amenities_content2)))
    # list2 = [im_amenities_content_dict[items] for items in list1]
    # list3 = list(set(list2).intersection(amen_list))
    # list4 = [x for x in list2 if x not in list3]
    # fin_amen_list = '-'.join(list4)
    # if len(list4) > 0:
    #     comment = 'Amenities which are not present in this property:' + str(fin_amen_list)
    #     model_comments.append(comment)

    quantity_score = 0
    # try:
    #     if room_quant_sc[hotel_id][0] == 0:
    #         if room_ct >= 4:
    #             quantity_score += 2
    #         else:
    #             model_comments.append('Min Room criteria is not satisfied')
    #     elif room_quant_sc[hotel_id][0] >= 0:
    #         if room_quant_sc[hotel_id][1] > 0:
    #             quantity_score += room_quant_sc[hotel_id][1] * 2
    #         else:
    #             model_comments.append('Min Room criteria in mrc is not satisfied')
    # except:
    #     quantity_score = 0
    #     model_comments.append('No Room photos are present')
    # if facade_ct >= 2:
    #     quantity_score += 2
    # else:
    #     model_comments.append('Min Facade criteria is not satisfied')
    # if wash_ct >= 2:
    #     quantity_score += 2
    # else:
    #     model_comments.append('Min Washroom criteria is not satisfied')
    # if recep_ct >= 2:
    #     quantity_score += 0.5
    # else:
    #     model_comments.append('Min Reception criteria is not satisfied')
    # if lobby_ct >= 2:
    #     quantity_score += 0.5
    # else:
    #     model_comments.append('Min Lobby criteria is not satisfied')
    # if misc_ct >= 2:
    #     quantity_score += 1
    # else:
    #     model_comments.append('Min Others criteria is not satisfied')
    # if len(images1) >= 35:
    #     quantity_score += 2
    # else:
    #     model_comments.append('Min images criteria is not satisfied')
    # try:
    #     if len(to_remove) > 0:
    #         comment = 'Out of total images ' + str(len(to_remove)) + ' were similar images and ' + str(
    #             len(images1)) + ' were evaluated.'
    #         model_comments.append(comment)
    # except:
    #     pass
    # try:
    #     del to_remove
    # except:
    #     pass

    #print('checkpoint 3 completed')

    aes_score1 = round(0.4 * 10 * aes_score, 2)
    angle_score1 = round(0.01 * 10 * angle_score, 2)
    comp_score1 = round(0.01 * 10 * comp_score, 2)
    dist_score1 = round(0.01 * 10 * dist_score, 2)
    st_score1 = round(0.01 * 10 * st_score, 2)
    lscape_score1 = round(0.01 * 10 * lscape_score, 2)
    aes_plus = round(float(
        np.where((aes_score1 + angle_score1 + comp_score1 + dist_score1 + st_score1 + lscape_score1) > 28,
                 28 * 1.6,
                 (aes_score1 + angle_score1 + comp_score1 + dist_score1 + st_score1 + lscape_score1) * 1.6)), 2)
    tech_score1 = round(0.2 * 10 * tech_score, 2)
    cont_score1 = round(0.0075 * 10 * cont_score, 2)
    white_score1 = round(0.0075 * 10 * white_score, 2)
    sat_score1 = round(0.0075 * 10 * sat_score, 2)
    sharp_score1 = round(0.0075 * 10 * sharp_score, 2)
    bright_score1 = round(0.0075 * 10 * bright_score, 2)
    exp_score1 = round(0.0075 * 10 * exp_score, 2)
    bwhite_score1 = round(0.005 * 10 * bwhite_score, 2)
    tech_plus = round(float(np.where((
                                             tech_score1 + cont_score1 + white_score1 + sat_score1 + sharp_score1 + bright_score1 + exp_score1 + bwhite_score1) > 15,
                                     15 * 1.6, (
                                             tech_score1 + cont_score1 + white_score1 + sat_score1 + sharp_score1 + bright_score1 + exp_score1 + bwhite_score1) * 1.6)),
                      2)
    tech_exif_score1 = round(0.1 * 10 * tech_exif_score, 2)
    quantity_score1 = round(0.2 * 10 * quantity_score, 2)
    final_score = round((aes_plus + tech_plus + tech_exif_score1 + quantity_score1), 2)
    overall = str("hotel_id") + ',' + str(len(images1)) + ',' + str(aes_score1) + ',' + str(
        angle_score1) + ',' + str(
        comp_score1) + ',' + str(dist_score1) + ',' + str(st_score1) + ',' + str(lscape_score1) + ',' + str(
        aes_plus) + ',' + str(tech_score1) + ',' + str(cont_score1) + ',' + str(white_score1) + ',' + str(
        sat_score1) + ',' + str(sharp_score1) + ',' + str(bright_score1) + ',' + str(exp_score1) + ',' + str(
        bwhite_score1) + ',' + str(tech_plus) + ',' + str(tech_exif_score1) + ',' + str(
        quantity_score1) + ',' + str(final_score) + ',' + '|'.join(model_comments)
    final.append(overall)
    final1 = pd.DataFrame(final)
    # final1.to_csv(mypath2 + country + "_" + version + '_prop_score_.csv', index=False)

    #print(final)

    try:
        img_score_propwise = pd.DataFrame()
        img_score_propwise['image_name'] = images1
        img_score_propwise['aes'] = round(aes_tech_data['aes_mean_grad'], 2)
        img_score_propwise['tech'] = round(aes_tech_data['tech_mean_grad'], 2)
        img_score_propwise['angle'] = round(aes_tech_data['angle'], 2)
        img_score_propwise['comp'] = round(aes_tech_data['comp'], 2)
        img_score_propwise['dist'] = round(aes_tech_data['dist'], 2)
        img_score_propwise['cont'] = round(aes_tech_data['cont'], 2)
        img_score_propwise['white'] = round(aes_tech_data['white'], 2)
        img_score_propwise['sat'] = round(aes_tech_data['sat'], 2)
        img_score_propwise['sharp'] = round(aes_tech_data['sharp'], 2)
        img_score_propwise['st'] = round(aes_tech_data['st'], 2)
        img_score_propwise['bright'] = round(aes_tech_data['bright'], 2)
        img_score_propwise['exp'] = round(aes_tech_data['exp'], 2)
        img_score_propwise['aesMean'] = round(img_score_propwise['aes'], 2)
        img_score_propwise['techMean'] = round(img_score_propwise['tech'], 2)
        img_score_propwise['exifScore'] = tech_exif_score  # TODO: NR: exif_score was throwing error
        img_score_propwise['aesPlus'] = round((4 / 4.5) * img_score_propwise['aes'] + (5 / (4 * 4.5)) * (
                img_score_propwise['angle'] + img_score_propwise['comp'] + img_score_propwise['dist'] +
                img_score_propwise['st']), 2)
        img_score_propwise['techPlus'] = round((2 / 2.5) * img_score_propwise['tech'] + (5 / (6 * 2.5)) * (
                img_score_propwise['white'] + img_score_propwise['sat'] + img_score_propwise['sharp'] +
                img_score_propwise['exp'] + img_score_propwise['bright'] + img_score_propwise['cont']), 2)
        img_score_propwise['aesMean'] = np.where(img_score_propwise['aesMean'] > 10, 10,
                                                  img_score_propwise['aesMean'])
        img_score_propwise['techMean'] = np.where(img_score_propwise['techMean'] > 10, 10,
                                                   img_score_propwise['techMean'])
        img_score_propwise['exifScore'] = np.where(img_score_propwise['exifScore'] > 10, 10,
                                                    img_score_propwise['exifScore'])
        img_score_propwise['imageScore'] = round(
            (4.5 / 8) * img_score_propwise['aesPlus'] + (2.5 / 8) * img_score_propwise['techPlus'] + (1 / 8) *
            img_score_propwise['exifScore'], 2)
        img_score_propwise = img_score_propwise[
            ['image_name', 'aesMean', 'angle', 'comp', 'dist', 'st', 'aesPlus', 'techMean',
             'white',
             'cont', 'sat', 'sharp', 'bright', 'exp', 'techPlus', 'exifScore', 'imageScore']]

        d = img_score_propwise.to_dict(orient='records')
        j = json.dumps(d)
        j = json.loads(j)
        j = j[0]
        l = []
        for key in j:
            l.append(key)
        encoded_string = map(str, l)
        b = {}
        for key in encoded_string:
            b[key] = j[key]
        print(b)
    except:
        pass

if __name__ == '__main__':
    total_start_time = datetime.now()
    main()
    total_end_time = datetime.now()
