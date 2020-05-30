import cv2
import numpy as np
import math

#define the screen resulation
screen_res = 1280, 720


def nothing(x):
    # any operation
    pass

capture  = cv2.VideoCapture(0)


#Parametres de Canny
cv2.namedWindow("Canny")
cv2.resizeWindow("Canny", 300, 100)
cv2.createTrackbar("Lower threshold", "Canny", 150, 255, nothing)
cv2.createTrackbar("Upper threshold", "Canny", 255, 255, nothing)

#Paramètres HSV : Hue(Teinte), Saturation, Value
cv2.namedWindow("Parametres HSV")
cv2.resizeWindow("Parametres HSV", 300, 300)
cv2.createTrackbar("Lower-H", "Parametres HSV", 0, 180, nothing)
cv2.createTrackbar("Lower-S", "Parametres HSV", 0, 255, nothing)
cv2.createTrackbar("Lower-V", "Parametres HSV", 123, 255, nothing)
cv2.createTrackbar("Upper-H", "Parametres HSV", 174, 180, nothing)
cv2.createTrackbar("Upper-S", "Parametres HSV", 134, 255, nothing)
cv2.createTrackbar("Upper-V", "Parametres HSV", 208, 255, nothing)

while True:
    success, img = capture.read()
    width = int(img.shape[1])
    height = int(img.shape[0])

    #on converti l'image en noir et blanc
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # select and ROI with user specified dimensions
    roi = imgGray[int(height/5):int(height-(height/5)),0:width]
    #on floute l'image (pour régler le flou, modifier (x,x)
    imgGrayBlur = cv2.GaussianBlur(roi, (5,5), 0)
    # on convertit l'image en HSV 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #on recupere la potition des curseurs HSV
    l_h = cv2.getTrackbarPos("Lower-H", "Parametres HSV")
    l_s = cv2.getTrackbarPos("Lower-S", "Parametres HSV")
    l_v = cv2.getTrackbarPos("Lower-V", "Parametres HSV")
    u_h = cv2.getTrackbarPos("Upper-H", "Parametres HSV")
    u_s = cv2.getTrackbarPos("Upper-S", "Parametres HSV")
    u_v = cv2.getTrackbarPos("Upper-V", "Parametres HSV")

    lower_red = np.array([l_h, l_s, l_v])
    upper_red = np.array([u_h, u_s, u_v])

    #on crée le masque
    mask = cv2.inRange(hsv, lower_red, upper_red)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)

    #contours a partir du masque HSV
    edges = cv2.Canny(mask, 100, 200)

    #Meilleure méthode : threshold adaptatif
    imgThreshAdapt = cv2.adaptiveThreshold(imgGrayBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,4)
    #on inverse les couleurs
    imgThreshAdapt = cv2.bitwise_not(imgThreshAdapt, imgThreshAdapt)

    

    #on détecte les lignes
    lines = cv2.HoughLinesP(imgThreshAdapt, 1, np.pi/180, 60, np.array([]), 50, 5)
    # iterate over the output lines and draw them
    plusGrandeLigne = 0
    x1Grand=0
    x2Grand=0 
    y1Grand=0 
    y2Grand=0
    for x in range (0, len(lines)):
        for x1, y1, x2, y2 in lines[x]:
            #print("test")
            pts = np.array([[x1, y1+int(height/5) ], [x2 , y2+int(height/5)]], np.int32)
            angleLigne = (math.atan2(x2-x1, (y2+int(height/5))-(y1+int(height/5))))*(180/math.pi)
            #on dessine la ligne
            
            if((math.sqrt(math.pow(x1-x2,2)+math.pow(y1-y2,2))) > plusGrandeLigne):
                plusGrandeLigne = math.sqrt(math.pow(x1-x2,2)+math.pow(y1-y2,2))
                x1Grand = x1
                x2Grand = x2
                y1Grand = y1
                y2Grand = y2
                angleGrandeLigne = (math.atan2(x2Grand-x1Grand, (y2Grand+int(height/5))-(y1Grand+int(height/5))))*(180/math.pi)
                #cv2.line(imgThreshAdapt, (x1, y1), (x2, y2), (255, 0, 0), 3)
            if(math.isclose(angleLigne, angleGrandeLigne, abs_tol = 1)):
                cv2.polylines(img, [pts], True, (0,255,0))
            print(angleLigne, angleGrandeLigne)
        #(x1Grand, y1Grand, x2Grand, y2Grand) = cv2.boundingRect(contour)
        
    
        cv2.line(img, (x1Grand, y1Grand+int(height/5)), (x2Grand, y2Grand+int(height/5)), (255, 0, 0), 2)
    


    
    cv2.imshow("Image originale", img)
    #cv2.imshow("Image floutee noir et blanc", imgGray)
    #cv2.imshow("Masque", mask)
    #cv2.imshow("Contours grace au masque", edges)
    cv2.imshow("Threshold adaptatif Gaussien", imgThreshAdapt)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



capture.release()
cv2.destroyAllWindows()