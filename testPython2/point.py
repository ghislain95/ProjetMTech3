import cv2
import numpy as np
import math

#define the screen resulation
screen_res = 1280, 720


def nothing(x):
    # any operation
    pass

capture  = cv2.VideoCapture(0)
_, img = capture.read()
old_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

params = dict(winSize = (15, 15))
maxLevel = 4
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)


# Mouse function
def select_point(event, x, y, flags, params):
    global point, point_selected, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True
        old_points = np.array([[x, y]], dtype=np.float32)

cv2.namedWindow("Image originale")
cv2.setMouseCallback("Image originale", select_point)

point_selected = False
point = ()
old_points = np.array([[]])

while True:
    _, img = capture.read()
    width = int(img.shape[1])
    height = int(img.shape[0])

    #on converti l'image en noir et blanc
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # select and ROI with user specified dimensions
    roi = imgGray[int(height/5):int(height-(height/5)),0:width]
    #on floute l'image (pour régler le flou, modifier (x,x)
    imgGrayBlur = cv2.GaussianBlur(roi, (13,13), 0)
    # on convertit l'image en HSV 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if point_selected is True:
        cv2.circle(img, point, 5, (0, 0, 255), 2)

        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, imgGray, old_points, None, **params)
        old_gray = imgGray.copy()
        old_points = new_points

        x, y = new_points.ravel()
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)


    niveau1 = cv2.pyrDown(img)
    #contours a partir du masque HSV
    #edges = cv2.Canny(mask, 100, 200)

    #Meilleure méthode : threshold adaptatif
    imgThreshAdapt = cv2.adaptiveThreshold(imgGrayBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    #on inverse les couleurs
    imgThreshAdapt = cv2.bitwise_not(imgThreshAdapt, imgThreshAdapt)

    


    


    
    cv2.imshow("Image originale", img)
    cv2.imshow("Niveau 1", niveau1)
    #cv2.imshow("Image floutee noir et blanc", imgGray)
    #cv2.imshow("Masque", mask)
    #cv2.imshow("Contours grace au masque", edges)
    cv2.imshow("Threshold adaptatif Gaussien", imgThreshAdapt)
    #cv2.imshow("Sans background", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



capture.release()
cv2.destroyAllWindows()