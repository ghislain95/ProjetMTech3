import cv2
import numpy as np
import math
from scipy.ndimage.interpolation import rotate



def nothing(x):
    # any operation
    pass

cap = cv2.VideoCapture(0)
img = cv2.imread("testGuitare.jpg")

framewidth=640
frameheight=480
#cap=cv2.VideoCapture(0)
cap.set(3,framewidth)
cap.set(4,frameheight)

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H", "Trackbars", 30, 180, nothing)
cv2.createTrackbar("L-S", "Trackbars", 40, 255, nothing)
cv2.createTrackbar("L-V", "Trackbars", 67, 255, nothing)
cv2.createTrackbar("U-H", "Trackbars", 73, 180, nothing)
cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U-V", "Trackbars", 255, 255, nothing)

font = cv2.FONT_HERSHEY_COMPLEX


(x0,y0,w0,h0)=(0,0,0,0)
(x0Petit, y0Petit) = (0,0)



numeroFretteCaisse = 12
nb = numeroFretteCaisse+1



def dessNote(x, y, grilleX, grilleY, img):
    return cv2.circle(img, (int(grilleX[y, x]), int(grilleY[y, x])), 2, (0, 0, 255), -1)


#Matrice de pourcentages : M[i] = longueur Sillet-ième frette en pourcentage (100% = longueurDiapason)
def matFr():
    M = np.linspace(0, 1, nb)
    for i in range (1,nb):
        M[i] = M[i-1]+((1-M[i-1])/(17.817))
    
    return M



def createGrid(x11,y11,x21,y21, M, longD):
    #crée un tableau allant de la valeur x21 à x11, en 13 points
    x = np.linspace(x21, int(x11), nb)#x[13] = 12eme frette
    y = np.linspace(int(y21), int(y11), 8)
    
    #on décale le x pour avoir les bonnes distances inter-frettes
    for i in range (0, nb):
        x[i] = x21 - int(M[i] * longD)
    
    #meshgrid permet de créer  deux tableaux à 2 dimensions
    x_1, y_1 = np.meshgrid(x, y)

    coordGrid = np.array([x_1, y_1])

    return coordGrid

def createGridX(x11,y11,x21,y21, M, longD):
    #crée un tableau allant de la valeur x21 à x11, en 13 points
    x = np.linspace(int(x21), int(x11), nb)
    y = np.linspace(int(y21), int(y11), 8)
    #x = np.linspace(int(y21), int(y11), 8)

    
   
    for i in range (0, nb):
        x[i] = x21 - int(M[i] * longD)
    

    #meshgrid permet de créer  deux tableaux à 2 dimensions
    x_1, y_1 = np.meshgrid(x, y)

    return x_1

def createGridY(x11,y11,x21,y21, M, longD):
    #crée un tableau allant de la valeur y21 à y11, en 13 points
    x = np.linspace(int(x21), int(x11), nb)
    y = np.linspace(int(y21), int(y11), 8)

    
    for i in range (0, nb):
        x[i] = x21 - int(M[i] * longD)
    
    #x = np.linspace(512,219,13)
    #y = np.linspace(210,326,8)

    #meshgrid permet de créer  deux tableaux à 2 dimensions
    x_1, y_1 = np.meshgrid(x, y)

    return y_1


#calcule coordonnées d'un point autour d'un autre
def rotatePoint(centerPoint,point,angle):
    angle = math.radians(angle)
    temp_point = point[0]-centerPoint[0] , point[1]-centerPoint[1]
    temp_point = ( temp_point[0]*math.cos(angle)-temp_point[1]*math.sin(angle) , temp_point[0]*math.sin(angle)+temp_point[1]*math.cos(angle))
    temp_point = temp_point[0]+centerPoint[0] , temp_point[1]+centerPoint[1]
    return temp_point

#créer une matrice dont tous les points sont tournés autour d'un point, d'un certain angle
#permet de tourner la matrice de points et de lui faire suivre le manche
def rotateGridX(grid, gridX, center, angleLigne):
    gridXR = gridX.copy()
    for x in range (0,nb):
        for y in range (0,7):
            point = grid[:, y, x]
            pointR = rotatePoint(center,point,90-angleLigne)
            gridXR[y,x] = pointR[0]
    return gridXR

def rotateGridY(grid, gridY, center, angleLigne):
    gridYR = gridY.copy()
    for x in range (0,nb):
        for y in range (0,7):
            point = grid[:, y, x]
            pointR = rotatePoint(center,point,90-angleLigne)
            gridYR[y,x] = pointR[1]
    return gridYR

while True:
    _, img = cap.read()
    width = int(img.shape[1])
    height = int(img.shape[0])
    imgBlur = cv2.GaussianBlur(img, (3,3), 0)
    hsv = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "Trackbars")

    #Idéal pour du vert / bleu:
    #l_h = 30    
    #l_s = 40
    #l_v = 67
    #u_h = 98 / 104
    #u_s = 255
    #u_v = 255

    lower_red = np.array([l_h, l_s, l_v])
    upper_red = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)
    test = mask.copy()



    #on cherche les contours dans le masque
    contours,hierarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    
    angleLigne=0
    (moyX1, moyX2, moyY1, moyY2) = (0,0,0,0)
    (sommeX1,sommeX2, sommeY1, sommeY2) = (0,0,0,0)
    (nbX1, nbX2, nbY1, nbY2) =(0,0,0,0)
    longManche = 0
    (x11,x12,x21,x22)=(0,0,0,0)
    (y11,y12,y21,y22)=(0,0,0,0)


    #on va définir les différents x et y de référence :
    for cnt in contours:
        #cv2.drawContours(img, contours, -1, (0,255,0),3)
        area=cv2.contourArea(cnt)
        Area_seuil=cv2.getTrackbarPos("Area", "Parameters")
        if ((area> (Area_seuil/50)) & (area>100)):
            cv2.drawContours(img, contours, -1, (0,255,0),3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            #print("\napprox : ", len(approx))
            x, y, w, h = cv2.boundingRect(approx)
            #print("x1 = ",x,"y1 = ",y)
            
            if((x < (width/2))):
                #cv2.rectangle(img, (x , y ), (x + w , y + h ), (0, 255, 0), 5)
                nbX1 += 1
                nbY1 += 1
                sommeX1 += x
                sommeY1 += y
                moyX1 += int(sommeX1/nbX1)
                moyY1 += int(sommeY1/nbY1)
                #print("moyX1 = ", moyX1, "moyY1 = ", moyY1)
            if(x > (width/2)):
                #cv2.rectangle(img, (x , y ), (x + w , y + h ), (0, 255, 0), 5)
                nbX2 += 1
                nbY2 += 1
                sommeX2 += x
                sommeY2 += y
                moyX2 += int(sommeX2/nbX2)
                moyY2 += int(sommeY2/nbY2)
                #print("moyX2 = ", moyX2, "moyY2 = ", moyY2)
            if(math.isclose(moyX2, x, abs_tol = 10)):
                x21 = x
                y21 = y
                
                x22 = x+w
                y22 = y+h
                
                cv2.line(img, (x21, y21), (x22, y22), (0, 0, 255),5)
                cv2.line(img, (x11, y11), (x21, y21), (0, 0, 255),5) #ligne manche
                angleLigne = (math.atan2(x21-x11, y21-y11))*(180/math.pi)
                
            tx = x22-moyX2
            ty = y22-y21
            Mat = [[1,0,tx],[0,1,ty]]
         
            if(math.isclose(moyX1, x, abs_tol = 10)):
                x11 = x+w+2
                y11 = y+int(h/2)+2
                
                x12 = x11 + tx
                y12 = y11 + ty
                
    #   -   -   -
    #   (x11,y11) : point situé à l'intersection manche-caisse
    #   (x21,y21) : point situé au sillet, corde 0 (mi grave)        
    #   (x22,y22) : point situé au sillet, corde 6 (mi aigu)        
    #   -   -   -           
             
    cv2.circle(img, (x11, y11), 5, (255, 0, 0), -1)
    cv2.circle(img, (x21, y21), 5, (255, 0, 0), -1)
    cv2.circle(img, (x22, y22), 5, (255, 0, 0), -1)

    longManche = math.sqrt(math.pow(x21-x11,2)+math.pow(y21-y11,2))          
    #print("longueur manche: ", longManche)
    largSillet = math.sqrt(math.pow(x22-x21,2)+math.pow(y22-y21,2))
    #print("largeur Sillet: ", largSillet)

    #on part du principe que le manche va de la frette 0 à la frette 12

    if(longManche == 0):
        longManche = 200
    #la 12eme frette se trouve à la moitié 
    longDiapason = (longManche*2)

    #matrice de pourcentages pour les distances inter-frettes
    M = matFr()
    for i in range (0,nb):
        print(M[i])



    #on crée les matrices de coordonnées pour les notes
    #en X : on va du sillet à la caisse
    grid = createGrid(int(x11),int(y22+((y22-y21)/3)),int(x21),int(y21), M, longDiapason)
    gridX = createGridX(int(x11),int(y22+((y22-y21)/3)),int(x21),int(y21), M, longDiapason)
    gridY = createGridY(int(x11),int(y22+((y22-y21)/3)),int(x21),int(y21), M, longDiapason)

    #centre de rotation : (x21,y21) = frette 0 corde 0 = grid[:,0,0]
    center = grid[:, 0, 0]

    #on rotationne les matrices X et Y, autour de center, suivant l'angle du manche
    gridXR = rotateGridX(grid, gridX, center, angleLigne)
    gridYR = rotateGridY(grid, gridY, center, angleLigne)

    x0 = int(gridX[0,0])
    y0 = int(gridY[0,0])
    #coordonnées de (x21,y21) (centre de rotation)
    print("\n-    Centre de rotation   -x0, y0 : ",x0, y0,"\n")

    for x in range(0,nb):
        for y in range(0,7):
            #(xG,yG) = coordonnées dans la grille avant rotation
            xG = gridX[y,x]
            yG = gridY[y,x]
            #(xR,yR) = coordonnées dans la grille après rotation
            xR = gridXR[y,x]
            yR = gridYR[y,x]

            #print("\n-    -   -xG, yG : ",xG, yG,"\n")
            print("\n-    -   -xR, yR : ",xR, yR,"\n")
            
            #on affiche chaque point de la matrice
            cv2.circle(img, (int(xR), int(yR)), 2, (0, 255, 0), -1)

    
    
    xTest = gridX[3,7]
    yTest = gridY[3,7]
    #La : 3eme corde 7eme case 
    print(grid[:, 3, 7])
    #dessiner une note a la 5eme case, 3eme corde
    dessNote(5,3,gridXR, gridYR, img)
    dessNote(4,4,gridXR, gridYR, img)
           
    print(angleLigne)



    cv2.imshow("Image", img)
    #cv2.imshow("Mask", mask)
    cv2.imshow("Test", test)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()