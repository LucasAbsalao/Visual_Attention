from matplotlib import pyplot as plt
import numpy as np
import cv2

def showAndDestroy(name, img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyWindow(name)

def plot_img(images, titles):
    fig,axes = plt.subplots(nrows=1,ncols=len(images), figsize=(15,15))
    for i,img in enumerate(images):
        axes[i].imshow(img)
        axes[i].set_title(titles[i])

    plt.show()

def exhibit(lines, columns, title, img, index=None):
  if index is None:
    if len(title) == len(img):
      index = np.arange(0,len(img)) + 1
  print(len(title))
  print(len(img))
  if len(index) == len(title) == len(img):
    #f,axes = plt.subplots(figsize=(15,15))
    for cont in range(len(index)):
      plt.subplot(lines,columns,index[cont])
      plt.title(title[cont])
      plt.xticks([]), plt.yticks([])
      plt.imshow(img[cont],cmap="gray")
    plt.show()
  else:
    print('Erro ao Exibir imagens')
