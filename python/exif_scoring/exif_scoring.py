from PIL import Image
import PIL

from python.utils.image_helper_utils import image_color_detection, landscape

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

def exif_scoring(images1, mypath):
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

        return exif_score, tech_exif_score, null_count, flag_dim_v1, flag_dim_v2, flag_iso_v1, flag_iso_v2, flag_res_v1, flag_res_v2, flag_ape_v1, flag_ape_v2, bwhite, lscape