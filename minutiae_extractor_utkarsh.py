## Code creater: NP

import fingerprint_enhancer
import fingerprint_feature_extractor
import cv2 as cv
from pathlib import Path

# def feature_vector(img0):
#     img1 = skeletonize(binarize(enhancer(img0)))
#     mask = filter_mask(img0)
#     minutiae_pts_arr = minutiae_generator(img1)
#     filtered_minutiae = list(filter(lambda m: mask[m[1], m[0]] > 20, minutiae_pts_arr))
#     minutiae_angle = minutiae_angles(filtered_minutiae, img1)
#     return minutiae_angle

def feature_vector(img0, thres=20):
    out = fingerprint_enhancer.enhance_Fingerprint(img0)
    FeaturesTerminations, FeaturesBifurcations = fingerprint_feature_extractor.extract_minutiae_features(out, spuriousMinutiaeThresh=thres, invertImage=False, showResult=False, saveResult=False)
    # make a list of minutiae X Y Type Angle
    fv = []
    for m in FeaturesTerminations+FeaturesBifurcations:
        fv.append([int(m.locX), int(m.locY), m.Type == "Termination", float(m.Orientation[0])])

    return fv


# def extract_minutiae(ip_path, op_path):
#     img = cv.imread(ip_path, cv.IMREAD_GRAYSCALE)
#     fv = feature_vector(img)
#     f = open(op_path, "w")
#     f.write(str(fv))


def extract_minutiae_vector(data):
    # from pymongo import MongoClient as mc
    try:
        ip_path, thres=data
    except Exception as e:
        print(str(e))
        ip_path, thres=data, 20
    # if not mc("10.5.18.101")["BI"]["mv"].count_documents({'path':ip_path},limit=1):
    ip_path=Path(ip_path).as_posix()
    img = cv.imread(ip_path, cv.IMREAD_GRAYSCALE)
    try:
        fv = feature_vector(img, thres)
    except Exception as e:
        print(str(e))
        fv=[]

    # set ip_path to relative path from folder starting with "FVC200", by recursively removing the parent folder
    def get_parent_path(p):
        if p.name == '' or p.name.startswith('FVC200'):
            return p.parent
        else:
            return get_parent_path(p.parent)
    ip_path = Path(ip_path).relative_to(get_parent_path(Path(ip_path))).as_posix()

    # if ip_path.startswith('D:/FVC Fingerprint Datasets/'):
    #     ip_path=ip_path.replace('D:/FVC Fingerprint Datasets/',"")
    
    doc={"path": ip_path, f"mv_{thres}": fv}
    return doc
