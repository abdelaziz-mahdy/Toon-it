import sys
import matplotlib as plt
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import tkinter.filedialog
from tkinter.filedialog import askopenfilename # Open dialog box
from PIL import Image
import cv2, time
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte,img_as_float
from skimage import data, io, filters
from matplotlib.pyplot import imshow, show, subplot, title, get_cmap
from skimage.exposure import rescale_intensity
from skimage.filters import laplace ,sobel, roberts
from scipy import ndimage
from scipy.fftpack import fft , fft2 ,fftshift , ifftshift , ifft2
from skimage.filters import gaussian
from skimage.filters import threshold_otsu
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage.morphology import disk
from PIL import Image, ImageTk, ImageDraw
import time
import math
import numba
from numba import njit, prange,vectorize,float32,cuda
from concurrent import futures






##function for creating the mean kernal
def meanFillterKernalcreator(size):
    kernal=[]
    element=1/(size*size)
    for x in range(size):
        kernal.append([])
        for y in range(size):
            kernal[x].append(element)
    return kernal



######## convolution ##############
@njit(fastmath=True)
def sumOfmatmuljitted(kernal,image):
    sum=0
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            sum+=kernal[x][y]*image[x][y]
    return sum
def sumOfmatmul(kernal,image):
    sum=0
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            sum+=kernal[x][y]*image[x][y]
    return sum
@njit(parallel=True, fastmath=True)
def convolutionLoop_parallel(image,image_padded,kernalx,kernaly,kernel):
    output = np.zeros_like(image)
    # Loop over every pixel of the image
    for x in prange(image.shape[0]):
        for y in range(image.shape[1]):
            output[x][y] = sumOfmatmuljitted(kernel,image_padded[x: x+kernalx,y: y+kernaly])
    return output

@njit(fastmath=True)
def convolutionLoop(image,image_padded,kernalx,kernaly,kernel):
    output = np.zeros_like(image)
    # Loop over every pixel of the image
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            output[x][y] = sumOfmatmuljitted(kernel,image_padded[x: x+kernalx,y: y+kernaly])
    return output

@adapt_rgb(each_channel)
def convolution(image, kernel,optimized=True,multicores=True):
    kernel=np.array(kernel)
    kernalx=len(kernel)
    kernaly=len(kernel[0])

    
    #Pads with the reflection of the vector mirrored along the edge of the array.
    image_padded = np.pad(image, ((kernalx//2), (kernaly//2)), 'symmetric')
    
    try:
        image_padded[((kernalx//2)):-((kernalx//2)), ((kernaly//2)):-((kernaly//2))] = image
    except:
        image_padded[((kernalx//2)+1):-((kernalx//2)-1), ((kernaly//2)+1):-((kernaly//2)-1)] = image
   
    if optimized:
        if multicores:
            return convolutionLoop_parallel(image,image_padded,kernalx,kernaly,kernel)
        else:
            return convolutionLoop(image,image_padded,kernalx,kernaly,kernel)
    else:
        # Loop over every pixel of the image
        output = np.zeros_like(image)
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                output[x][y]=sumOfmatmul(kernel,image_padded[x: x+kernalx,y: y+kernaly])
        return output
######## end of convolution ##############


########## median #####################
@njit(parallel=True, fastmath=True)
def medianLoopJitted_parallel(image,image_padded,kernalx,kernaly,kernel):
    output = np.zeros_like(image)
    # Loop over every pixel of the image
    for x in prange(image.shape[0]):
        for y in range(image.shape[1]):
            output[x][y]=np.median((image_padded[x: x+kernalx,y: y+kernaly]))
    return output

@njit(fastmath=True)
def medianLoopJitted(image,image_padded,kernalx,kernaly,kernel):
    output = np.zeros_like(image)
    # Loop over every pixel of the image
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            output[x][y]=np.median((image_padded[x: x+kernalx,y: y+kernaly]))
    return output

def medianLoop(image,image_padded,kernalx,kernaly,kernel):
    output = np.zeros_like(image)
    # Loop over every pixel of the image
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            output[x][y]=np.median((image_padded[x: x+kernalx,y: y+kernaly]))
    return output
@adapt_rgb(each_channel)
def median(image, kernel,optimized=True,multicores=True):
    kernel=np.array(kernel)
    kernalx=len(kernel)
    kernaly=len(kernel[0])

    # Add zero padding to the input image (lead to black border)
    #image_padded = np.zeros((image.shape[0] + (kernalx-1), image.shape[1] + (kernaly-1)))
    
    
    #Pads with the reflection of the vector mirrored along the edge of the array.
    image_padded = np.pad(image, ((kernalx//2), (kernaly//2)), 'symmetric')
    
    try:
        image_padded[((kernalx//2)):-((kernalx//2)), ((kernaly//2)):-((kernaly//2))] = image
    except:
        image_padded[((kernalx//2)+1):-((kernalx//2)-1), ((kernaly//2)+1):-((kernaly//2)-1)] = image
   
    if optimized:
        if multicores:
            return medianLoopJitted_parallel(image,image_padded,kernalx,kernaly,kernel)
        else:
            return medianLoopJitted(image,image_padded,kernalx,kernaly,kernel)
    else:
        output = np.zeros_like(image)
        # Loop over every pixel of the image
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                output[x][y]=medianLoop((image_padded[x: x+kernalx,y: y+kernaly]).ravel())

        return output
##########end of median#####################



#####quantizationImageColors######
@njit(parallel=True, fastmath=True)
def quantizeloopjitted_parallel(image,ratio):
    #ratio is quantization ratio
    for i in prange(image.shape[0]):
        for j in range(image.shape[1]):
            image[i][j]=round(image[i][j]*(ratio/255))*(255/ratio);
    return image
@njit(fastmath=True)
def quantizeloopjitted(image,ratio):
    #ratio is quantization ratio
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i][j]=round(image[i][j]*(ratio/255))*(255/ratio);
    return image

@adapt_rgb(each_channel)
def quantizationImageColors(image,ratio,optimized=True,multicores=True):
   
    if optimized:
        if multicores:
            return quantizeloopjitted_parallel(image,ratio)
        else:
            return quantizeloopjitted(image,ratio)
    else:
        #ratio is quantization ratio
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i][j]=round(image[i][j]*(ratio/255))*(255/ratio);
    return image
##########End of quantizationImageColors #####################



#rgb to gray
def rgb2grayMY(rgb):
    if(len(rgb.shape)==3):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return rgb




########## Dilation ########################
@njit(fastmath=True)
def maxOfmatmuljitted(kernal,image):
    max=0
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            mul=kernal[x][y]*image[x][y]
            if max<mul:
                max=mul
    return max


def maxOfmatmul(kernal,image):
    max=0
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            mul=kernal[x][y]*image[x][y]
            if max<mul:
                max=mul
    return max

@njit(parallel=True, fastmath=True)
def dilationLoop_parallel(image,image_padded,kernalx,kernaly,kernel):
    output = np.zeros_like(image)
    # Loop over every pixel of the image
    for x in prange(image.shape[0]):
        for y in range(image.shape[1]):
            output[x][y] = maxOfmatmuljitted(kernel,image_padded[x: x+kernalx,y: y+kernaly])
    return output

@njit(fastmath=True)
def dilationLoop(image,image_padded,kernalx,kernaly,kernel):
    output = np.zeros_like(image)
    # Loop over every pixel of the image
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            output[x][y] = maxOfmatmuljitted(kernel,image_padded[x: x+kernalx,y: y+kernaly])
    return output

@adapt_rgb(each_channel)
def dilationBYME(image, kernel,optimized=True,multicores=True):

    kernalx=len(kernel)
    kernaly=len(kernel[0])
    # convolution output
    output = np.copy(image)
    
    
    #Pads with the reflection of the vector mirrored along the edge of the array.
    image_padded = np.pad(image, ((kernalx//2), (kernaly//2)), 'symmetric')
    
    if optimized:
        if multicores:
            return dilationLoop_parallel(image,image_padded,kernalx,kernaly,kernel)
        else:
            return dilationLoop(image,image_padded,kernalx,kernaly,kernel)

    else:
        # Loop over every pixel of the image
        output = np.zeros_like(image)
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                output[x][y]=maxOfmatmul(kernel,image_padded[x: x+kernalx,y: y+kernaly])
    return output
##########end of Dilation ########################


#set progress in the ui and update it 
def setProgress(step):
    timeProc.set("%s/9" %step)
    rw.update()

def checkIfinvertisNeeded(edges):
    unique, counts = np.unique(edges, return_counts=True)
    dic=dict(zip(unique, counts))
    if(dic[0]>dic[1]):
        return True
    else:
        return False
## cartonize image
def cartonize(img,colors,blurValue,optimized=True,multicores=True):
    #convert to float to make all the functions use same image type
    img=np.float64(img)
    setProgress(1)
    #convert to gray to get clearer edges
    gray=rgb2grayMY(img)
    setProgress(2)
    #apply median filter to reduce image edges 
    blurred=convolution(gray,meanFillterKernalcreator(blurValue),optimized,multicores)
    setProgress(3)
    #convolution with laplace filter to get edges
    edges=convolution(blurred,[[0,-1,0],[-1,4,-1],[0,-1,0]],optimized,multicores)
    setProgress(4)
    #reduce edges shaprness
    edges=convolution(edges,meanFillterKernalcreator(blurValue),optimized,multicores)
    setProgress(5)
    #thersholding edges to be easier in adding to original
    edges=edges >= threshold_otsu(edges)

    #check if the image needs a converting to background white and edges black
    setProgress(6)
    if checkIfinvertisNeeded(edges):
        edges=np.invert(edges)

    #reduce edges but we removed it for better edges
    edges=dilationBYME(edges,disk(1),optimized,multicores)


    setProgress(7)

    blurredrgb=convolution(img,meanFillterKernalcreator(blurValue),optimized,multicores)

    #quantization Image Colors 
    setProgress(8)
    blurredrgb=quantizationImageColors(blurredrgb,colors,optimized,multicores)
    
    edges=np.uint8(edges)
    blurredrgb=np.uint8(blurredrgb)
    setProgress(9)
    final=cv2.bitwise_and(blurredrgb, blurredrgb, mask=edges)
    return final

#same as cartonize but witout the steps and no return (used to compile optimized code to have the best performance)
def compile(img,colors,optimized=True,multicores=True):
    img=np.float64(img)
    gray=rgb2grayMY(img)
    blurred=convolution(gray,meanFillterKernalcreator(2),optimized,multicores)
    edges=convolution(blurred,[[0,-1,0],[-1,4,-1],[0,-1,0]],optimized,multicores)
    edges=convolution(edges,meanFillterKernalcreator(2),optimized,multicores)
    edges=edges >= threshold_otsu(edges)
    edges=dilationBYME(edges,disk(1),optimized,multicores)
    blurredrgb=convolution(img,meanFillterKernalcreator(2),optimized,multicores)
    blurredrgb=quantizationImageColors(blurredrgb,colors,optimized,multicores)
    edges=np.uint8(edges)
    blurredrgb=np.uint8(blurredrgb)
    final=cv2.bitwise_and(blurredrgb, blurredrgb, mask=edges)







############## UI functions ######################
def getCartVal():
    return sc.get()

def getBlurVal():
    return sc2.get()

def selection():
    return option.get()

def removeGif():
    load.config(text="")
    caned.delete("all")
    caned.create_image(0,0, anchor=NW, image=img)
    caned.create_image(1000,0, anchor=NE, image=img2)

def SaveImg():
    files = [('Image', '*.png')]
    file = filedialog.asksaveasfile(filetypes = files, defaultextension = files,initialfile = "cartoned")
    #fileN = filedialog.asksaveasfile(mode='w', defaultextension=".png")
    if file:
        #im.save(file) # saves the image to the input file name. 
        cv2.imwrite(file.name,cartoned)



def applyCartonize():
    caned.delete("all")
    global Resimg
    global img, load, cartoned##, loading, loading2,frames  
    img = ImageTk.PhotoImage(Resimg)
    loading = Image.open("loading.png")
    loading = loading.resize((50, 50), Image.ANTIALIAS)
    loader = ImageTk.PhotoImage(loading)
    caned.create_image(500,250, anchor=CENTER, image= loader)
    load = Label(caned, text='Loading...') ##Put the PNG loading image using canvas.create_image above the label
    load.place(x=470,y=280)
    rw.update()  
    if(selection() == 1):
        start_time = time.time()
        cartoned = cartonize(cv2.imread(filename),getCartVal(),getBlurVal(),optimized=False,multicores=False) # Since most of the work done on Image is on NP Arrays and not Pillow Image 
        end_time =int(round(time.time() - start_time * 100))
        timeProc.set("%s seconds" % (time.time() - start_time))
    elif(selection()==2):
        start_time = time.time()                                                                        #object the images was opened using CV and it returns np arras
        cartoned = cartonize(cv2.imread(filename),getCartVal(),getBlurVal(),optimized=True,multicores=False)
        end_time =int(round(time.time() - start_time * 100))
        timeProc.set("%s seconds" % (time.time() - start_time))
    else:
        start_time = time.time()
        cartoned = cartonize(cv2.imread(filename),getCartVal(),getBlurVal(),optimized=True,multicores=True)
        end_time =int(round(time.time() - start_time * 100))
        timeProc.set("%s seconds" % (time.time() - start_time))
         
    dims = (470,520)
    cartonedIMAGE = cv2.resize (cartoned,dims)
    global img2
    cartonedIMAGE = cv2.cvtColor(cartonedIMAGE, cv2.COLOR_BGR2RGB)
    cartonedIMAGE = Image.fromarray(cartonedIMAGE)
    img2 =  ImageTk.PhotoImage(image=cartonedIMAGE) ## Then the image is transformed back to Image using Image.fromarray()to display to the user. 
    removeGif()
    rw.update() 


def compilingAtStartUp():
    timeProc.set("compiling functions ")
    rw.update()
    try:
        filename="compiling image.png"
        timeProc.set("compiling functions 1/2")
        rw.update()
        compile(cv2.imread(filename),3,optimized=True,multicores=False)
        timeProc.set("compiling functions 2/2")
        rw.update()
        compile(cv2.imread(filename),3,optimized=True,multicores=True)
    except:
        print("compiling image.png is not avaliable will compile when running code")
    timeProc.set("")
    rw.update()

def openImage():
    global filename
    types = ('*.png','*.jpeg','*.jpg')
    tmp = askopenfilename(filetypes=[("images",types)])##
    #if its selected
    if tmp!="":
        caned.delete("all")
        filename=tmp
        global Resimg
        Resimg = Image.open(filename)
        Resimg = Resimg.resize((470, 520), Image.ANTIALIAS) 
        applyCartonize()

rw = Tk()                    # Create window
thread_pool_executor = futures.ThreadPoolExecutor(max_workers=1) #thread queue
option = IntVar()
option.set(3)
timeProc = StringVar()
rw.title("Toon-It!")     
rw.geometry("1000x650") # Setting window dimensions
rw.resizable(FALSE,FALSE) 
caned = Canvas(rw, width = 1000, height = 500) # Reserving space for images to show  
caned.pack()
lab1=Label(rw, text="Select an image to Cartonize IT")
lab1.place(x=430,y=530)
lab3=Label(rw, text="Image Proccessed In:")
lab3.place(x=25,y=540)
lab4 = Label(rw, textvariable=timeProc)
lab4.place(x=150,y=540) 
butn1=Button(rw,text="Browse images", width =17)
butn1.place(x=450,y=550)
butn1.config(command=lambda: [thread_pool_executor.submit(openImage)]) #Once an image is selected from the file explorer the image will be cartoonized and will show the original image on the right
butn2=Button(rw,text="Apply Cartonize", width =17)
butn2.place(x=850,y=570)
butn2.config(command=lambda: [thread_pool_executor.submit(applyCartonize)])
butn3 = Button(rw, text="Saved Cartonized Image", width = 17)
butn3.place(x=850,y=600)
butn3.config(command=lambda: [thread_pool_executor.submit(SaveImg)])

sc = Scale(rw, from_ = 1, to=25, orient=HORIZONTAL)
sc.set(4)
sc.place(x=450,y=580)
sc2 = Scale(rw, from_ = 2, to=25, orient=HORIZONTAL)
sc2.set(4)
sc2.place(x=250,y=580)
lab2 = Label(rw, text="Colors Scale:")
lab2.place(x=370,y=598)
lab3= Label(rw, text="Blurring Scale:")
lab3.place(x=169,y=598)

rad1 = Radiobutton(rw, text="Non-optimized (slow)", variable=option, value=1) #Radiobutton for returning values to use the few options set infront of the user
rad1.place(x=600,y=530)
rad2 = Radiobutton(rw, text="Optimized (faster)", variable=option, value=2)
rad2.place(x=600,y=560)
rad3 = Radiobutton(rw, text="Optimized With Multicores (fastest)", variable=option, value=3)
rad3.place(x=600,y=590)
thread_pool_executor.submit(compilingAtStartUp)
rw.mainloop()
