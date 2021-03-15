
#Append Comments to Image as per its Individual models' score
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