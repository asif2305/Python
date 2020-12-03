# source :https://www.youtube.com/watch?v=WQeoO7MI0Bs&t=549s&ab_channel=Murtaza%27sWorkshop-RoboticsandAI
import cv2
import numpy as np

# chapter 1
'''
# img=cv2.imread("Image/lena.jpg")

# cv2.imshow("Output",img)
# cv2.waitKey(0)

# image show from vedios
#cap = cv2.VideoCapture("Image/Dis.mp4")
# Vedio show using webcam
cap=cv2.VideoCapture(0) # for 1 webcam
cap.set(3,640) # width with id 3
cap.set(4,480) # height with id 4
# increase brightness
cap.set(5,100)

while True:
    success, img = cap.read()  # success is boolean
    cv2.imshow("Vedio", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


'''
# chapter 2 :Basic function
img = cv2.imread("Image/lena.jpg")
kernel = np.ones((5, 5), np.uint8)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Gray image
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)  # 7,7 is the kernel size
# canny edge detector
imgCanny = cv2.Canny(img, 150, 200)  # 100 is threshold value
# increase the thickness of the image
imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)
# opposite of dialation
imgEroded = cv2.erode(imgDialation, kernel, iterations=1)

# cv2.imshow("Gray Image", img)
# cv2.imshow("Blur Image", imgBlur)
# cv2.imshow("Canny Image", imgCanny)
# cv2.imshow("Dialation Image", imgDialation)
# cv2.imshow("Eroded Image", imgEroded)

# chapter 3:Resizing and Cropping

imgLambo = cv2.imread("Image/lambo.jpg")
print(imgLambo.shape)

imgResize = cv2.resize(imgLambo, (1000, 500))  # x,y
print(imgResize.shape)
# specific part of image

imgCropped = imgLambo[0:200, 200:500]

# cv2.imshow("Original Image",imgLambo)
# cv2.imshow("Resize Image",imgResize)
# cv2.imshow("Cropped Image",imgCropped)

# chapter 4 :Shapes and text

imgShape = np.zeros((512, 512, 3), np.uint8)
# print(imgShape)
# imgShape[:]=255,0,0
# draw line
cv2.line(imgShape, (0, 0), (imgShape.shape[1], imgShape.shape[0]), (0, 255, 0), 3)
cv2.line(imgShape, (0, 0), (300, 300), (0, 255, 0), 3)

cv2.rectangle(imgShape, (0, 0), (250, 350), (0, 0, 255), 2)
cv2.circle(imgShape, (400, 50), 30, (255, 255, 0), 5)

cv2.putText(imgShape, "OPENCV", (300, 500), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 150, 0), 3)

# cv2.imshow("Shape Image",imgShape)

# chapter 5: warp prespective

imgWarp = cv2.imread("Image/card.jpg")
cv2.imshow("Card Image", imgWarp)

# define four corner of the card
width, height = 250, 450
pts1 = np.float32([[111, 219], [287, 188], [154, 482], [352, 440]])
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgOutput = cv2.warpPerspective(imgWarp, matrix, (width, height))

cv2.imshow("Card Image", imgWarp)
cv2.imshow("Card Image Output", imgOutput)

cv2.waitKey(0)
