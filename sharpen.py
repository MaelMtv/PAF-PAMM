from PIL import Image, ImageFilter
import cv2


filename = "LAB_equa_1.jpg" # put the name of the image to sharpen, best if it is already equalized
cv2.imshow('before', cv2.imread(filename,1))
with Image.open(filename) as img:
    img.load()

sharp = img.filter(ImageFilter.SHARPEN)
sharp.show()

#sharp = sharp.save('LAB_equa_1_sharp.jpg') #saves the sharpened image in a new file that you specify the name

cv2.waitKey(0)
cv2.destroyAllWindows()