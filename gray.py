import matplotlib.pyplot as plt
import cv2

image = cv2.imread("tumeur.jpg",1)



def mean(img,i,j):
    return(img[i][j][0]/3+img[i][j][1]/3+img[i][j][2]/3)
    
def luma(img,i,j):
    return(0.3*img[i][j][0]+0.59*img[i][j][1]+0.11*img[i][j][2])

def luma2(img,i,j):
    return(0.2126*img[i][j][0]+0.7152*img[i][j][1]+0.0722*img[i][j][2])

def luma3(img,i,j):
    return(0.299*img[i][j][0]+0.587*img[i][j][1]+0.114*img[i][j][2])
        
def desaturation(img,i,j):
    ma = max(img[i][j][0],img[i][j][1],img[i][j][2])
    mi = min(img[i][j][0],img[i][j][1],img[i][j][2])
    return(ma/2+mi/2)

def decompositionmax(img,i,j):
    return(max(img[i][j][0],img[i][j][1],img[i][j][2]))

def decompositionmin(img,i,j):
    return(min(img[i][j][0],img[i][j][1],img[i][j][2]))

def grayred(img,i,j):
    return(img[i][j][2])

def graygreen(img,i,j):
    return(img[i][j][1])

def grayblue(img,i,j):
    return(img[i][j][0])


def convert_gray(n, img): #choose n from 0 to 9
    assert 0 <= n < 10
    new=[]
    for i in range(len(img)):
        new.append([])
        for j in range(len(img[0])):
            if n==0:
                m = mean(img,i,j)
                #plt.title("mean")
            elif n==1:
                m= luma(img,i,j)
                #plt.title("luma")
            elif n==2:
                m= luma2(img,i,j)
                #plt.title("luma2")
            elif n==3:
                m= luma3(img,i,j)
                #plt.title("luma3")
            elif n==4:
                m= desaturation(img,i,j)
                #plt.title("desaturation")
            elif n==5:
                m= decompositionmax(img,i,j)
                #plt.title("decomposition max")
            elif n==6:
                m= decompositionmin(img,i,j)
                #plt.title("decomposition min")
            elif n==7:
                m= grayred(img,i,j)
                #plt.title("red")
            elif n==8:
                m= graygreen(img,i,j)
                #plt.title("green")
            elif n==9:
                m= grayblue(img,i,j)
                #plt.title("blue")
            new[i].append(m)
    return new


def compare(): # Compares all the above implemented method to convert RGB to grayscale by displaying them
    fig = plt.figure()
    for i in range(10):
        fig.add_subplot(4,3,i+1)
        gimage = convert_gray(i,image)
        plt.imshow(gimage, cmap = 'gray')
    plt.show()

#compare()

"""Next section compares OpenCV with the above code using the same conversion formula"""
mea = convert_gray(3,image)
good = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.gray()
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(mea)
fig.add_subplot(1,2,2)
plt.imshow(good)
plt.show()