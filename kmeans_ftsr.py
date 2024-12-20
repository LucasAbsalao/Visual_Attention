import cv2
import numpy as np 
import math
from matplotlib import pyplot as plt
import os
from show import showAndDestroy, plot_img, exhibit
from colorSystem import rgb2lab

def lab_mean(img):
    L,a,b = cv2.split(img)
    titles = ["L","a","b"]
    #exhibit(1,3,titles,[L,a,b])
    Lmean = np.mean(L)
    amean = np.mean(a)
    bmean = np.mean(b)

    Imean = np.array([Lmean, amean, bmean])
    
    return Imean


def gaussian_smoothing(img, kernel):
    #result = cv2.filter2D(img,-1,kernel)
    result = cv2.GaussianBlur(img,(5,5),0)
    return result


def euclidean_distance(vector1, vector2):
    vector = (vector1-vector2)
    #print(vector1, vector2, vector)
    soma = np.float32(0)
    for x in vector:
        soma += x**2
    #print(math.sqrt(soma))
    return math.sqrt(soma)
    #return np.linalg.norm(vector)

def saliency_map(img, Imean):
    img32f = np.float32(img)
    Imean32f = np.float32(Imean)

    Iwhc = np.zeros((img32f.shape[0],img32f.shape[1]), dtype = np.float32)
    for i in range(Iwhc.shape[0]):
        for j in range(Iwhc.shape[1]):
            Iwhc[i][j] = euclidean_distance(Imean32f,img32f[i][j])
    
    #Iwhc2 = [[euclidean_distance(Imean32f,element) for element in row] for row in img32f]
    Iwhc = np.array(np.round(Iwhc), dtype=np.uint8)
    return Iwhc

def k_means(img):
    vectorized = img.reshape((-1,3))
    vectorized = np.float32(vectorized)

    k=6
    attempts = 10
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.95)

    ret, label, center = cv2.kmeans(vectorized,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))

    return result_image

def segment_mean(img):
    return 2*(np.uint8(np.round(np.mean(img))))

def get_unique_values(img):
    only = np.unique(img.reshape(-1,3), axis=0)
    return only

def segment_saliency_map(segmented_img, saliency, unique):
    segmented_saliency = np.zeros(saliency.shape, dtype=np.uint8)
    for uni in unique:
        saliency_values=[]
        saliency_index=[]
        for i, row in enumerate(segmented_img):
            for j,element in enumerate(row):
                if element.tolist() == uni.tolist():
                    saliency_index.append((i,j))
                    saliency_values.append(saliency[i][j])

        media = np.uint8(np.round(np.mean(saliency_values)))

        for (i,j) in saliency_index:
            segmented_saliency[i,j] = media

    return segmented_saliency 

def write_images(path_folder,name_folder,images):
    os.system('mkdir ' + path_folder)
    write_img = []
    for image in images:
        write_img.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    cv2.imwrite(path_folder+name_folder+'_imagem_original.png', write_img[0])
    cv2.imwrite(path_folder+name_folder+'_saliency_map.png', write_img[1])
    cv2.imwrite(path_folder+name_folder+'_threshold_otsu.png', write_img[2])
    cv2.imwrite(path_folder+name_folder+'_kmeansHSV_saliency_map.png', write_img[3])
    cv2.imwrite(path_folder+name_folder+'_kmeans_image.png', write_img[4])

    
PATH_IMAGES = 'images/'
PATH_OUTPUT = 'kmeans_hsv_saliency/'
path = PATH_IMAGES + 'Tool_10.png'

name_folder = (path.split('/')[-1]).split('.')[0]
path_folder = PATH_OUTPUT + name_folder + '/'

img = cv2.imread(path)

#img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_lab = rgb2lab(img_rgb)
Imean = lab_mean(img_lab)

#Figura que representa a média das features do vetor no espaço Lab TODO: Conversão de Lab para rgb
'''Imean_rgb = cv2.cvtColor(np.array([[Imean]]), cv2.COLOR_LAB2RGB)
img_mean = np.full((img_lab.shape), Imean_rgb, dtype = np.uint8)
plt.imshow(img_mean)
plt.show()'''

print(img.shape)


#Smoothing da imagem com gaussiana 5x5 e criação do mapa de saliência
kernel=np.array([1,4,6,4,1])/16  
img_lab_smooth = gaussian_smoothing(img_lab,kernel)
saliency_img = saliency_map(img_lab_smooth,Imean)

plt.imshow(saliency_img, cmap='gray')
plt.show()
print("receba ", Imean, img_lab_smooth[img_lab_smooth.shape[0]-1][img_lab_smooth.shape[1]-4], saliency_img[img_lab_smooth.shape[0]-1][img_lab_smooth.shape[1]-4])

#Imagem gerada utilizando a técnica de OTSU
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
ret, thresh = cv2.threshold(saliency_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
attention_img_otsu = cv2.bitwise_and(img, img, mask=thresh)


plt.figure(figsize=(6.4*5,4.8*5), constrained_layout=False)
titles = ['Imagem Original', 'Mapa de Saliência', 'Imagem Final Otsu']
images = [img, saliency_img, attention_img_otsu]
exhibit(1,3,titles,images)


#Imagem gerada usando Kmeans
segment_hsv = k_means(img_lab)
segment_rgb = cv2.cvtColor(segment_hsv, cv2.COLOR_LUV2RGB)
segments_values = get_unique_values(segment_rgb)


segment_saliency = segment_saliency_map(segment_rgb, saliency_img, segments_values)
segment_m = segment_mean(saliency_img)
print(segment_m)
ret, thresh = cv2.threshold(segment_saliency, segment_m, 255, cv2.THRESH_BINARY)
attention_img_kmeans = cv2.bitwise_and(img,img,mask=thresh)


plt.figure(figsize=(6.4*5,4.8*5), constrained_layout=False)
titles = ['Imagem Original', 'Segmentação Kmeans', 'Mapa de Saliência', 'Segmentação Saliencia','Imagem Final']
images = [img, segment_rgb,saliency_img,segment_saliency,attention_img_kmeans]
exhibit(1,5,titles,images)

images = [img, saliency_img, attention_img_otsu, segment_saliency,attention_img_kmeans]
write_images(path_folder, name_folder,images)