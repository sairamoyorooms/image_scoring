from __future__ import print_function
import json
a = {u'cont': 1.0, u'tech_plus': 2.85, u'aes_mean': 2.15, u'comp': 1.0, u'exif_score': 0.0, u'tech_mean': 1.38, u'aes_plus': 3.02, u'image_name': u'_aab9037a45518738.jpg.jpg', u'bright': 0.88, u'exp': 0.86, u'sharp': 1.0, u'angle': 1.0, u'white': 0.49, u'st': 1.0, u'image_score': 2.59, u'dist': 1.0, u'sat': 1.0}

l=[]

for key in a:
    l.append(key)

encoded_string = map(str, l)

b = {}

for key in encoded_string:
    b[key] = a[key]

print(b)