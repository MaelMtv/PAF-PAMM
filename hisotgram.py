import matplotlib.pyplot as plt
import cv2


image = cv2.imread("test1.jpg",1)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#image = cv2.merge((gray_image,gray_image,gray_image)) #comment to keep a colored image

def histogram_gray(img): #computes the histogram and the cumulative histogram of a grayscale image
    a,b = img.shape
    total = a*b
    y=[0]*256
    for i in range(len(img)):
        for j in img[i]:
            n = round(j)
            y[n] += 1
    z = [x/total for x in y]
    z2 = [0]*256
    z2[0] = z[0]
    for l in range(1,256):
        z2[l]=z[l]+z2[l-1]
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(z, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.plot(z2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


cv2.imshow('init',image)



"""LAB space equalization"""
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l,a,b = cv2.split(lab_image)

#histogram_gray(l)

equa = cv2.equalizeHist(l)
#histogram_gray(equa)

updated_lab_image = cv2.merge((equa,a,b))
hist_eq_image = cv2.cvtColor(updated_lab_image, cv2.COLOR_LAB2BGR)
cv2.imshow('LABinter',hist_eq_image) # displays equalized image
#cv2.imwrite("LAB_equa_4.jpg", hist_eq_image) # saves normal equalization result in the file of your choice

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
clahe_image = clahe.apply(l)
#histogram_gray(clahe_image)

updated_lab_image2 = cv2.merge((clahe_image,a,b))
final_image = cv2.cvtColor(updated_lab_image2, cv2.COLOR_LAB2BGR)
cv2.imshow('LABfinal',final_image) # displays image after clahe algorithm
#cv2.imwrite("LAB_clahe_4.jpg",final_image) # saves clahe equalization result in the file of your choice

"""HSV space equalization"""
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv_image)

#histogram_gray(v)

equa = cv2.equalizeHist(v)
#histogram_gray(equa)

updated_hsv_image = cv2.merge((h,s,equa))
hist_eq_image = cv2.cvtColor(updated_hsv_image, cv2.COLOR_HSV2BGR)
cv2.imshow('HSVinter',hist_eq_image) # displays equalized image
#cv2.imwrite("HSV_equa_4.jpg", hist_eq_image) # saves normal equalization result in the file of your choice

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
clahe_image = clahe.apply(v)
#histogram_gray(clahe_image)

updated_hsv_image2 = cv2.merge((h,s,clahe_image))
final_image = cv2.cvtColor(updated_hsv_image2, cv2.COLOR_HSV2BGR)
cv2.imshow('HSVfinal',final_image) # displays image after clahe algorithm
#cv2.imwrite("HSV_clahe_4.jpg", final_image) # saves clahe equalization result in the file of your choice

"""YUV space equalization"""
yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
y,u,v = cv2.split(yuv_image)

#histogram_gray(y)

equa = cv2.equalizeHist(y)
#histogram_gray(equa)

updated_yuv_image = cv2.merge((equa,u,v))
hist_eq_image = cv2.cvtColor(updated_yuv_image, cv2.COLOR_YUV2BGR)
cv2.imshow('YUVinter',hist_eq_image) # displays equalized image
#cv2.imwrite("YUV_equa_4.jpg", hist_eq_image) # saves normal equalization result in the file of your choice


clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
clahe_image = clahe.apply(y)
#histogram_gray(clahe_image)

updated_yuv_image2 = cv2.merge((clahe_image,u,v))
final_image = cv2.cvtColor(updated_yuv_image2, cv2.COLOR_YUV2BGR)
cv2.imshow('YUVfinal',final_image) # displays image after clahe algorithm 
#cv2.imwrite("YUV_clahe_4.jpg",final_image) # saves clahe equalization result in the file of your choice

cv2.waitKey(0)
cv2.destroyAllWindows()