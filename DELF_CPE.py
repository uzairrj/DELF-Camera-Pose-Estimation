import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.feature import plot_matches
import cv2 as cv
import tensorflow as tf
import tensorflow_hub as hub
import os
import gc
from sklearn.neighbors import KDTree
import scipy.io as sio
import scipy.io
import sys

def readReferenceData(refPath):
    cordinates = []
    with open(refPath,"r") as fp:
        lines = fp.readlines()
        for l in lines[2:]:
            l=l.replace("\n","")
            try:
                index,y,x = l.split(" ")
                cordinates.append((int(index),float(x),float(y)))
            except:
                print("Exception: ",l)
    
    return cordinates

def readMeshFile(refPath):
    cordinates = []
    with open(refPath,"r") as fp:
        lines = fp.readlines()
        for l in lines[14:]:
            l=l.replace("\n","")
            x,y,z = l.split(" ")[0:3]
            cordinates.append((float(x),float(y), float(z)))
    return np.array(cordinates)

def readImageAndVisibilityData(img_path, img_visibility, width=256, height=256):
    cordinates = np.array(readReferenceData(img_visibility))
    image = Image.open(img_path)
    image = np.asarray(image)
    #image = cv.imread(img_path).copy()
    _max = max(cordinates[:,2])
    _min = min(cordinates[:,2])                                                                                                         
    #cordinates[:,2] = image.shape[0] - cordinates[:,2]
  
    img = image.copy()
    #img = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    for c in cordinates[:]:
        img = cv.circle(img, (int(c[2]),int(c[1])), radius=10, color=(106, 255, 0), thickness=-1)

    fig,ax = plt.subplots(1,2)
    fig.suptitle(os.path.basename(img_path))
    ax[0].set_title("Orignal")
    ax[0].imshow(img)

    max_x = img.shape[1]
    max_y = img.shape[0]

    ratio_y = max_y/width
    ratio_x = max_x/height

    cordinates[:,2] = cordinates[:,2] / ratio_y
    cordinates[:,1] = cordinates[:,1] / ratio_x

    image_re = cv.resize(np.array(image),(height,width))
    #image_re = cv.cvtColor(image_re, cv.COLOR_RGB2BGR)
    img = image_re.copy()
    for c in cordinates[:]:
        img = cv.circle(img, (int(c[2]),int(c[1])), radius=5, color=(106, 255, 0), thickness=-1)


    ax[1].set_title("Resized")
    ax[1].imshow(img)

    return {"image":image_re, "cordinates":cordinates}


def run_delf(image, delf):
    np_image = np.array(image)
    float_image = tf.image.convert_image_dtype(np_image, tf.float32)

    return delf(
      image=float_image,
      score_threshold=tf.constant(50.0),
      image_scales=tf.constant([0.00075,0.001,0.005,0.0075,0.01,0.03,0.05,0.075,0.10,0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 1.75, 2.0, 2.3]),
      max_feature_num=tf.constant(100000))

def filterDescResult(image_data, result,threshold):
    tree = KDTree(image_data["cordinates"][:,1:3], leaf_size=50, metric='euclidean')
    dist , indices = tree.query(result['locations'].numpy(), k=1)
    indices = indices.ravel()
    threshold_indices = np.array([i for i in range(len(dist)) if dist[i] < threshold ])
    filtered_indices = indices[threshold_indices]
    duplicate_index = [idx for idx, item in enumerate(filtered_indices) if item in filtered_indices[:idx]]
    mask = np.full(len(filtered_indices),True, dtype=bool)
    mask[duplicate_index] = False
    filtered_indices = filtered_indices[mask]
    threshold_indices = threshold_indices[mask]
    result_new = {}
    result_new['locations'] = result['locations'].numpy()[threshold_indices]
    result_new['descriptors'] = result['descriptors'].numpy()[threshold_indices]
    result_new['boxes'] = result['boxes'].numpy()[threshold_indices]
    return result_new, image_data["cordinates"][filtered_indices]

def convertCordinates(data,max_width, max_height, width,height, is_p3d = True):
    _data = data.copy()
    ratio_x = max_width/width
    ratio_y = max_height/height
    if is_p3d:
        _data[:,1] = _data[:,1]/ratio_x
        _data[:,2] = _data[:,2]/ratio_y
    else:
        _data[:,0] = _data[:,0]/ratio_x
        _data[:,1] = _data[:,1]/ratio_y
    return _data

def driver(model_img1_path,model_img1_vis, ref_img2_path,ref_img2_vis):
    point_3d_dict = scipy.io.loadmat('./helping data/3dpoints_data.mat')
    point3d = point_3d_dict["point3d"]
    
    image1_data = readImageAndVisibilityData(model_img1_path,model_img1_vis,2500,3750)
    image2_data = readImageAndVisibilityData(ref_img2_path,ref_img2_vis,2500,3750)
    
    delf = hub.load('https://tfhub.dev/google/delf/1').signatures['default']
    
    result1 = run_delf(image1_data["image"],delf)
    gc.collect()
    result2 = run_delf(image2_data["image"],delf)
    
    filtered_delf_result1,orignal_data1 = filterDescResult(image1_data, result1, 5)
    print("Interesting Points Found: ",len(filtered_delf_result1["locations"]))
    img = image1_data["image"].copy()

    for c in filtered_delf_result1["locations"][:]:
        img = cv.circle(img, (int(c[1]),int(c[0])), radius=17, color=(255, 0, 0), thickness=2) #Red

    for c in orignal_data1[:,1:3]:
        img = cv.circle(img, (int(c[1]),int(c[0])), radius=10, color=(0, 0, 255), thickness=-1) #Blue
    
    plt.imshow(img)
    
    img = image2_data["image"].copy()

    for c in result2["locations"][:]:
        img = cv.circle(img, (int(c[1]),int(c[0])), radius=17, color=(255, 0, 0), thickness=2)

    
    plt.imshow(img)
    
    
    num_features_1 = filtered_delf_result1['locations'].shape[0]
    print("Loaded image 1's %d features" % num_features_1)

    num_features_2 = result2['locations'].shape[0]
    print("Loaded image 2's %d features" % num_features_2)


    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

    matches = bf.match(np.array(filtered_delf_result1["descriptors"]),np.array(result2["descriptors"]))

    good = []
    locations_2_to_use = []
    locations_1_to_use = {"p2d":[],"p3d":[]}
    myMatches = []

    threshold = 0.60
    for m in matches:
        if m.distance < threshold:
            good.append(m)
            locations_1_to_use["p2d"].append(orignal_data1[m.queryIdx][1:3])
            locations_1_to_use["p3d"].append(point3d[int(orignal_data1[m.queryIdx][0])])
            locations_2_to_use.append(result2["locations"][m.trainIdx])
        
    print(len(good))
    myMatches = [[i,i] for i in range(len(locations_2_to_use))]
    myMatches = np.row_stack(myMatches)
    _, ax = plt.subplots()
    plot_matches(
          ax,
          image2_data["image"],
          image1_data["image"],
          np.array(locations_2_to_use),
          np.array(locations_1_to_use["p2d"]),
          np.array(myMatches),
          matches_color='b')
    ax.axis('off')
    ax.set_title('DELF correspon"dences')
    
    temp_2d_cordinates = convertCordinates(np.array(locations_1_to_use["p2d"]),image1_data["image"].shape[1],image1_data["image"].shape[0], 5472, 3648, False)

    model_image = {
        "p3d":locations_1_to_use["p3d"],
        "p2d":temp_2d_cordinates
    }
    
    data = convertCordinates(np.array(locations_2_to_use),image2_data["image"].shape[1],image2_data["image"].shape[0], 5472, 3648,False)
    ref_image = {
        "p2d":data,
    }
    
    model_base_name = os.path.basename(model_img1_path)
    model_filename_without_ext = os.path.splitext(model_base_name)[0]
    
    
    ref_base_name = os.path.basename(ref_img2_path)
    ref_filename_without_ext = os.path.splitext(ref_base_name)[0]

    sio.savemat(model_filename_without_ext+"_data"+".mat",model_image)
    sio.savemat(ref_filename_without_ext+"_data"+".mat",ref_image)

    print("\n**All Done!**")

if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("\nError: Invalid Arguments!!")
        print(f"\nUsage: {sys.argv[0]} <reference image> <reference visibility> <target image> <target visibility>")
        exit(1)

    driver(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])