#*****************************************************************************#
#    Программа детектирования людей на изображении на основе HOG-детектора    #
#										                                      #
#*****************************************************************************#


from __future__ import print_function
import datetime
import imutils
import numpy as np
import cv2
from imutils.object_detection import non_max_suppression

# Путь к обрабатываемому изображению
imagePath = "Resources/person_01.bmp"

# Настройки HOG-детектора
WinStride = (8, 8)
Padding = (16, 16)
Scale = 1.06
MeanShift = 0

# Настройки положения рамки
DeltaH = 0.9
DeltaW = 0.93
DeltaY = 0.05
DeltaX = 0.15

# Инициализация датчика людей
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Загрузка изображения и приведение размера
original_image = cv2.imread(imagePath)
start = datetime.datetime.now()
image = imutils.resize(original_image, width=min(400, original_image.shape[1]))

# Детектирование людей на изображении
start = datetime.datetime.now()
(rects, weights) = hog.detectMultiScale(image, hitThreshold=0, winStride=WinStride, padding=Padding, scale=Scale, useMeanshiftGrouping=MeanShift)
print("Humans were found: ", len(rects))
# Исключение перекрывающихся рамок
pick = non_max_suppression(rects, probs=None, overlapThresh=10)

# Отрисовка границ рамки
for(x, y, w, h) in pick:
    cv2.rectangle(image, (int(x+w*DeltaX), int(y+h*DeltaY)), (x + int(w*DeltaW), y + int(h*DeltaH)), (0, 255, 0), 1)
original_image = imutils.resize(image, width=original_image.shape[1])
print("Detection time: {} s".format(
    (datetime.datetime.now() - start).total_seconds()))
# Вывод изображения
cv2.imshow("Detections", original_image)
cv2.waitKey(0)
