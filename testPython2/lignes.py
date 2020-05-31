import cv2
import numpy as np
import math

#define the screen resulation
screen_res = 1280, 720


def nothing(x):
    # any operation
    pass

capture  = cv2.VideoCapture(0)


while True:
    success, img = capture.read()
    width = int(img.shape[1])
    height = int(img.shape[0])

    #on converti l'image en noir et blanc
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # select and ROI with user specified dimensions
    roi = imgGray[int(height/5):int(height-(height/5)),0:width]
    #on floute l'image (pour régler le flou, modifier (x,x)
    imgGrayBlur = cv2.GaussianBlur(roi, (3,3), 0)


    #Meilleure méthode : threshold adaptatif
    imgThreshAdapt = cv2.adaptiveThreshold(imgGrayBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    #on inverse les couleurs
    imgThreshAdapt = cv2.bitwise_not(imgThreshAdapt, imgThreshAdapt)

    #on détecte les lignes
    lines = cv2.HoughLinesP(imgThreshAdapt, 1, np.pi/180, 60, np.array([]), 50, 5)
    #paramètres utiles à la détection des bonnes lignes
    plusGrandeLigne = 0
    x1Grand=0
    x2Grand=0 
    y1Grand=0 
    y2Grand=0
    nbLignes=0
    moyenneX1 = 0
    moyenneX2 = 0
    moyenneY1 = 0
    moyenneY2 = 0
    sommeX1=0
    sommeY1=0
    sommeX2=0
    sommeY2=0

    #on boucle dans les lignes pour effectuer nos calculs
    for x in range (0, len(lines)):
        for x1, y1, x2, y2 in lines[x]:
            pts = np.array([[x1, y1+int(height/5) ], [x2 , y2+int(height/5)]], np.int32)
            angleLigne = (math.atan2(x2-x1, (y2+int(height/5))-(y1+int(height/5))))*(180/math.pi)
            
            #pour déterminer la plus grande ligne
            if((math.sqrt(math.pow(x1-x2,2)+math.pow(y1-y2,2))) > plusGrandeLigne):
                plusGrandeLigne = math.sqrt(math.pow(x1-x2,2)+math.pow(y1-y2,2))
                x1Grand = x1
                x2Grand = x2
                y1Grand = y1
                y2Grand = y2
                angleGrandeLigne = (math.atan2(x2Grand-x1Grand, (y2Grand+int(height/5))-(y1Grand+int(height/5))))*(180/math.pi)
                #cv2.line(imgThreshAdapt, (x1, y1), (x2, y2), (255, 0, 0), 3)
            #si les lignes sont un minimum parallèles avec la plus grande
            if(math.isclose(angleLigne, angleGrandeLigne, abs_tol = 1)):
                nbLignes+=1
                sommeX1 = sommeX1+ x1
                sommeY1 = sommeY1+ y1
                sommeX2 = sommeX2+ x2
                sommeY2 = sommeY2+ y2
                moyenneX1 = moyenneX1 + sommeX1/nbLignes
                moyenneY1 = moyenneY1 + sommeY1/nbLignes
                moyenneX2 = moyenneX2 + sommeX2/nbLignes
                moyenneY2 = moyenneY2 + sommeY2/nbLignes

                #on dessine la ligne
                cv2.line(img, (x1, y1+int(height/5)), (x2, y2+int(height/5)), (0, 255, 0), 2)
                #cv2.polylines(img, [pts], True, (0,255,0))
            print(angleLigne, angleGrandeLigne,nbLignes,moyenneX1)
        

        #cv2.line(img, (int(moyenneX1), int(moyenneY1)-int(height/5)), (int(moyenneX2), int(moyenneY2)-int(height/5)), (0, 0, 255), 2)
        #cv2.rectangle(img,(int(moyenneX1),-int(moyenneY1+int(height/5))),(int(moyenneX1+moyenneX2),-int(moyenneY1+int(height/5)+(moyenneY2+int(height/5)))),(0,0,255),1)
        #cv2.circle(img, (int(moyenneX1),int(moyenneY1+int(height/5))), radius=0, color=(0, 0, 255), thickness=10)
        cv2.line(img, (x1Grand, y1Grand+int(height/5)), (x2Grand, y2Grand+int(height/5)), (255, 0, 0), 2)
    

    cv2.imshow("Image originale", img)
    #cv2.imshow("Image floutee noir et blanc", imgGrayBlur)
    cv2.imshow("Threshold adaptatif Gaussien", imgThreshAdapt)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



capture.release()
cv2.destroyAllWindows()