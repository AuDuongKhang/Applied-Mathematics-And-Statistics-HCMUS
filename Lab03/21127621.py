from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def advanced(value):
    if (value < 0):
        return 0
    if (value > 255):
        return 255
    return value

#g(x,y) = co * f(x,y) + br
#change brightness -> co = 1
def change_brightness(img,img_new, br):
    pixel = img.load()
    pixel_new = img_new.load()
    for i in range(img_new.size[0]):
        for j in range(img_new.size[1]):
            r, b, g = pixel[i,j]
            _r = advanced(r + br)
            _b = advanced(b + br)
            _g = advanced(g + br)
            pixel_new[i,j] = (_r, _b, _g)
            
    return img_new

#change constrast -> br = 0
def change_contrast(img,img_new, co):
    pixel = img.load()
    pixel_new = img_new.load()
    for i in range(img_new.size[0]):
        for j in range(img_new.size[1]):
            r, b, g = pixel[i,j]
            _r = int(round(advanced(r*co)))
            _b = int(round(advanced(b*co)))
            _g = int(round(advanced(g*co)))
            pixel_new[i,j] = (_r, _b, _g)
            
    return img_new

def horizontal_flip(img):
    img = np.fliplr(img)
    return img

def vertical_flip(img):
    img = np.flipud(img)
    return img

#imgGray=0.2989 * r + 0.5870 * g + 0.1140 * b for all channels
def gray_scale(img,img_new):
    pixel = img.load()
    pixel_new = img_new.load()
    for i in range(img_new.size[0]):
        for j in range(img_new.size[1]):
            r, b, g = pixel[i,j]
            gr = int(round(advanced(0.2989 * r + 0.5870 * g + 0.1140 * b)))
            pixel_new[i,j] = (gr, gr, gr)
            
    return img_new

#red = 0.393 * r + 0.769 * g + 0.189 * b
#green = 0.349 * r + 0.686 * g + 0.168 * b
#blue = 0.272 * r + 0.534 * g + 0.131 * b
def sepia(img,img_new):
    pixel = img.load()
    pixel_new = img_new.load()
    for i in range(img_new.size[0]):
        for j in range(img_new.size[1]):
            r, b, g = pixel[i,j]
            sr = int(round(advanced(0.393 * r + 0.769 * g + 0.189 * b)))
            sg = int(round(advanced(0.349 * r + 0.686 * g + 0.168 * b)))
            sb = int(round(advanced(0.272 * r + 0.534 * g + 0.131 * b)))

            pixel_new[i,j] = (sr, sg, sb)
            
    return img_new

#print the image from the center -> center crop
def crop(img, size):
    img_new = img[int((img.shape[0]/2)-(size[0]/2)):int((img.shape[0]/2)+(size[0]/2)),int((img.shape[1]/2)-(size[1]/2)):int((img.shape[1]/2)+(size[1]/2))]
    return img_new

def circle_crop(img):
    img_new = np.zeros(img.shape)
    width = img.shape[0]
    height = img.shape[1]
    #radius of circle
    radius = min(width, height)//2
    #center of circle
    center = np.array([width//2,height//2])
    
    #check border of circle. True -> img, false ->black img
    for i in range (width):
        for j in range (height):
            distance_2 = (i-center[0])**2 +(j-center[1])**2
            if distance_2 > radius**2:
                img_new[i,j] = np.zeros(img.shape[2])
            else:
                img_new[i,j] = img[i,j] 
                
    return img_new

def double_ellipse_crop(img):
    # Create a blank mask array
    mask = np.zeros_like(img)

    # Define the parameters of the two ellipses
    center1 = (img.shape[1]//2, img.shape[0]//2) # Center of the first ellipse
    axes1 = (img.shape[1]//2, img.shape[0]//3) # Axes of the first ellipse
    angle1 = np.radians(45) # Angle of rotation of the first ellipse
    color1 = (255, 255, 255) # White color for the first ellipse

    center2 = (img.shape[1]//2, img.shape[0]//2) # Center of the second ellipse
    axes2 = (img.shape[1]//3, img.shape[0]//2) # Axes of the second ellipse
    angle2 = np.radians(45) # Angle of rotation of the second ellipse
    color2 = (255, 255, 255) # White color for the second ellipse

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            # Calculate the distance from the pixel to the center of each ellipse
            dx1 = j - center1[0]
            dy1 = i - center1[1]
            dx2 = j - center2[0]
            dy2 = i - center2[1]

            # Rotate the distance vector by the negative angle of each ellipse
            rx1 = dx1 * np.cos(-angle1) - dy1 * np.sin(-angle1)
            ry1 = dx1 * np.sin(-angle1) + dy1 * np.cos(-angle1)
            rx2 = dx2 * np.cos(-angle2) - dy2 * np.sin(-angle2)
            ry2 = dx2 * np.sin(-angle2) + dy2 * np.cos(-angle2)

            # Check if the rotated distance satisfies the equation of an ellipse
            if (rx1**2 / axes1[0]**2 + ry1**2 / axes1[1]**2 <= 1) or (rx2**2 / axes2[0]**2 + ry2**2 / axes2[1]**2 <= 1):
                # If yes, set the pixel value to the color of the ellipse
                mask[i][j] = color1

    # Apply the mask to the image
    masked_img = np.where(mask > 0, img, 0)

    return masked_img

    
def blur(img, kernel):
    img_new = np.zeros(img.shape)
    
    #Convolution
    for i in range (kernel.shape[0]):
        for j in range (kernel.shape[1]):
            rShiftValue = i - kernel.shape[0]//2
            cShiftValue = j - kernel.shape[1]//2
            shiftArray = np.roll(img,(rShiftValue, cShiftValue), axis=(0,1))
            img_new+= shiftArray * kernel[i,j] 
            
    return img_new  

def sharpen(img, kernel):
    img_new = np.zeros(img.shape, dtype=np.uint8)
    height, width, channels = img.shape
    k_height, k_width = kernel.shape[:2]

    # Normalize kernel
    kernel_sum = np.sum(kernel)
    if kernel_sum != 0:
        kernel = kernel / kernel_sum

    # Handle padding
    border = max(k_height // 2, k_width // 2)
    img_padded = np.pad(img, ((border, border), (border, border), (0, 0)), mode='edge')

    # Convolution
    for i in range(height):
        for j in range(width):
            for c in range(channels):
                img_new[i, j, c] = np.clip(np.sum(img_padded[i:i+k_height, j:j+k_width, c] * kernel), 0, 255)

    return img_new


def save_image(img, name, option, mode=0):
    new_img = Image.fromarray(img.astype(np.uint8))
    
    if option == 1:
        new_name = name[:-4] + "_brightness" + name[-4:]
        
    elif option == 2:
        new_name = name[:-4] + "_constrast" + name[-4:]
        
    elif option == 3:
        
        if mode == 0:
            new_name = name[:-4] + "_flip_vertical" + name[-4:]
        elif mode == 1:
            new_name = name[:-4] + "_flip_horizontal" + name[-4:]
                
    elif option == 4:
        
        if mode == 0:
            new_name = name[:-4] + "_grey" + name[-4:]
        elif mode == 1:
            new_name = name[:-4] + "_sepia" + name[-4:]
                       
    elif option == 5:
        
        if mode == 0:
            new_name = name[:-4] + "_blur" + name[-4:]
        elif mode == 1:
            new_name = name[:-4] + "_sharpen" + name[-4:]
                        
    elif option == 6:
        new_name = name[:-4] + "_center_crop" + name[-4:]
        
    elif option == 7:
        new_name = name[:-4] + "_circle_crop" + name[-4:]
    
    elif option == 8:
        new_name = name[:-4] + "_ellipse_crop" + name[-4:]  
        
    new_img.save(new_name)

def choose_option(img,name,option):
    if option == 1:
       img_new = Image.new(img.mode, img.size)
       brightness = int (input('Enter brightness: '))
       img_new = change_brightness(img,img_new,brightness)  
       img_new = np.asarray(img_new)
       save_image(img_new,name,option)
       
    elif option == 2:
       img_new = Image.new(img.mode, img.size)
       constrast = float (input('Enter your constrast: '))
       img_new = change_contrast(img,img_new,constrast)  
       img_new = np.asarray(img_new)
       save_image(img_new,name,option)
       
    elif option == 3:
        img = np.asarray(img)
        print('We have 2 modes of flip: ')
        print('0. Flip vertical')
        print('1. Flip horizontal')
        mode = int (input('Enter your mode: '))
        if mode == 0:
            img_new = vertical_flip(img)  
            
        elif mode == 1:
            img_new = horizontal_flip(img)  
        
        save_image(img_new,name,option,mode)    
    
    elif option == 4:
        img_new = Image.new(img.mode, img.size)
        print('We have 2 modes: ')
        print('0. Grey scale')
        print('1. Sepia tone')
        mode = int (input('Enter your mode: '))
        if mode == 0:
            img_new = gray_scale(img,img_new)  
        
        elif mode == 1:
            img_new = sepia(img,img_new)  
        
        img_new = np.asarray(img_new)
        save_image(img_new,name,option,mode)
    
    elif option == 5:
        img = np.asarray(img)
        print('We have 2 modes: ')
        print('0. Blur')
        print('1. Sharpen')
        mode = int (input('Enter your mode: '))
        if mode == 0:
            #kernel = np.array([[1,2,1], [2,4,2], [1,2,1]])/16
            kernel = np.array([[1,4,6,4,1], [4,16,24,16,4], [6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])/256
            img_new = blur(img,kernel)  
        
        elif mode == 1:
            kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
            img_new = sharpen(img,kernel)  

        save_image(img_new,name,option,mode)
    
    elif option == 6:
        img = np.asarray(img)
        sz = input('Enter size which you want to resize (Ex. 300x300): ')
        sz = sz.split("x")
        size = [int(sz[0]) , int(sz[1])] 
        img_new = crop(img,size)
        save_image(img_new,name,option)
        
    elif option == 7:
        img = np.asarray(img) 
        img_new = circle_crop(img)
        save_image(img_new,name,option)
    
    elif option == 8:
        img = np.asarray(img)
        img_new = double_ellipse_crop(img)
        save_image(img_new,name,option)    

if __name__ == "__main__":
    name  = input('Input file of name to compress + extension (E.G: test1.jpg): ')
    #2.png
    img = Image.open(name)
    print('Image processing ... ')
    print(' 1. Change Brightness')
    print(' 2. Change Constrast')
    print(' 3. Flip vertical/horizontal')
    print(' 4. RGB to gray/sepia')
    print(' 5. Blur/Sharpen')
    print(' 6. Crop image by size (crop in the center)')
    print(' 7. Circle crop ')
    print(' 8. Double elip crop ')
    print(' 0. All options  ')
    option = 0
    option = int(input('Enter your options: '))
    if option == 0:
        for i in range(1,9):
            choose_option(img,name,i)
    else: 
        choose_option(img,name,option)        
                            