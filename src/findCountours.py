import os
from pathlib import Path
import numpy as np
import cv2

"""
Take all images on a white page and extract them to seperate files.
Has threshold s. t. small things will be disregarded.

Need two folders ../import and ../export to work.
"""

dirImg = os.listdir(path='../import')
padding = 0
total = 0
loopNumber = 1

for imgName in dirImg:
    # Import Image
    img = cv2.imread('../import/'+imgName)
    deep_copy = img.copy()

    # Create dir
    Path("../export/"+imgName).mkdir(parents=True, exist_ok=True)


    """
    # Color tweaks for the original image.
    alpha = 1 # Contrast control (1.0-3.0)
    beta = 0 # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    # cv2.imshow ('Adjusted', adjusted)

    # Convert to grayscale.
    imgGray = cv2.cvtColor (adjusted, cv2.COLOR_BGR2GRAY)
    """

    imgGray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

    # GaussianBlur
    dst = cv2.GaussianBlur(imgGray,(3,3),cv2.BORDER_DEFAULT)

    # Threshold
    ret, thresh = cv2.threshold(dst, 213, 255, cv2.THRESH_BINARY)
    thresh = 255 - thresh

    # Contours
    threshold_area = 4000
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(deep_copy, contours, -1, (0, 255, 0), 3)
    cv2.imwrite('../export/'+imgName+'/'+imgName+'cnt.jpg', deep_copy)
    # cv2.imwrite('../export2/'+imgName+'/'+imgName+'gray.jpg', imgGray)
    cv2.imwrite('../export/'+imgName+'/'+imgName+'thres.jpg', thresh)

    # Put every detected contour that is bigger then 'threshold_area' in an
    # own file.
    i = 0
    for cnt in enumerate(contours):
        area = cv2.contourArea(contours[i])
        if area > threshold_area:
            # print (i)
            [x, y, width, height] = cv2.boundingRect(contours[i])

            # Create mask where white is what we want, black otherwise
            mask = np.zeros_like(img)
            # Draw filled contour in mask
            cv2.drawContours(mask, contours, i, (255,255,255), -1)
            # Extract out the object and place into output image
            out = np.ones_like(img) * 255
            out[mask == 255] = img[mask == 255]

            if (i == 0):
                roi = out[y:y+height, x:x+width]
            else:
                roi = out[y:y+height, x:x+width]

            name = '../export/'+imgName+'/'+imgName+'-' + str(i) + '.jpg'

            outputImage = cv2.copyMakeBorder(
                 roi,
                 padding,
                 padding,
                 padding,
                 padding,
                 cv2.BORDER_CONSTANT,
                 value=[255, 255, 255]
            )

            cv2.imwrite(name, outputImage)

        i = i + 1

    print('[' + str(loopNumber) + '/' + str(len(dirImg)) + '] ' + str(i) + ' - ' + imgName)
    loopNumber = loopNumber + 1
    total = total + i

print()
print('Processed ' + str(total) + ' edges in total!')
