import cv2
import numpy as np
import math
from scipy.ndimage.interpolation import rotate



def nothing(x):
    # any operation
    pass

cap = cv2.VideoCapture(0)
#img = cv2.imread("testGuitare.jpg")

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

cv2.namedWindow("Manche")
cv2.createTrackbar("largeur", "Manche", 98, 120, nothing)
cv2.createTrackbar("longueur", "Manche", 98, 120, nothing)
cv2.createTrackbar("hauteur", "Manche", 38, 100, nothing)
cv2.createTrackbar("Alignement", "Manche", 60, 100, nothing)
cv2.createTrackbar("Transparence", "Manche", 100, 500, nothing)

font = cv2.FONT_HERSHEY_COMPLEX

#################### IMPORTATION DE LA TEXTURE #############################
#path 
path = r'C:\Users\Arnaud\Desktop\Projet M&T\mache_21.png'
#path = "testGuit.jpg"
# Using cv2.imread() method
img_manche = cv2.imread(path)


# Displaying the image 
cv2.imshow('Texture', img_manche) 


in_alpha=1-0.4
############################################################################



#sur ma guitare, la frette au niveau de la caisse de résonance est la frette n°12
numeroFretteCaisse = 12
nb = numeroFretteCaisse+1

longManche = 0
longDiapason = 0
#on utilise distance1214 si la frette au niveau de la caisse n'est pas 12 mais 14
distance1214 = 0.1028571429 #distance entre la frette 12 et 14 en pourcentage : 1 = longueur du manche



def dessNote(x, y, grilleX, grilleY, img):
    return cv2.circle(img, (int(grilleX[y, x]), int(grilleY[y, x])), 2, (0, 0, 255), -1)


#Matrice de pourcentages : M[i] = longueur Sillet-ième frette en pourcentage (1 = longueurDiapason)
def matFr():
    M = np.linspace(0, 1, 20)
    for i in range (1,nb):
        M[i] = M[i-1]+((1-M[i-1])/(17.817))
    
    return M



def createGrid(x11,y11,x21,y21, M, longD):
    #crée un tableau allant de la valeur x21 à x11, en 13 points
    x = np.linspace(x21, int(x11), nb)#x[13] = 12eme frette
    y = np.linspace(int(y21), int(y11), 6)
    
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
    y = np.linspace(int(y21), int(y11), 6)
    #x = np.linspace(int(y21), int(y11), 8)

    
   
    for i in range (0, nb):
        x[i] = x21 - int(M[i] * longD)
    

    #meshgrid permet de créer  deux tableaux à 2 dimensions
    x_1, y_1 = np.meshgrid(x, y)

    return x_1

def createGridY(x11,y11,x21,y21, M, longD):
    #crée un tableau allant de la valeur y21 à y11, en 13 points
    x = np.linspace(int(x21), int(x11), nb)
    y = np.linspace(int(y21), int(y11), 6)

    
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
        for y in range (0,6):
            point = grid[:, y, x]
            pointR = rotatePoint(center,point,90-angleLigne)
            gridXR[y,x] = pointR[0]
    return gridXR

def rotateGridY(grid, gridY, center, angleLigne):
    gridYR = gridY.copy()
    for x in range (0,nb):
        for y in range (0,6):
            point = grid[:, y, x]
            pointR = rotatePoint(center,point,90-angleLigne)
            gridYR[y,x] = pointR[1]
    return gridYR





def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
 
def texture(x11,y11,x12,y12,x21,y21,x22,y22,angle):
    #-------------------------------------------------------------------------
    # Mauve (x11,y11)
    # Bleu(x21,y21)
    # Beige(x12,y12)
    # Rose(x22,y22)
    #détermine les dimensions de la du manche pour redimensioner la texture
    e= cv2.getTrackbarPos("largeur", "Manche")
    l= cv2.getTrackbarPos("longueur", "Manche")
    img_height=abs(y11-y12)+e
    img_width=abs(x21-x12) +l
    #dim = (img_width, img_height)
    #☻dim=img_manche.shape
    #print("shape", img_manche.shape)
    print("w",img_width,"h",img_height)
    #--------------------------------------------------------------------------
    
    
    
    
    if(img_width>=200 and img_height>=10):  # dimension minimal pour coller la texture
        #        print('Original Dimensions : ',img_manche.shape)
        #"correction", "Manche" // "fin", "Manche" // "longueur", "Manche"
        offset= cv2.getTrackbarPos("hauteur", "Manche")
        fin=cv2.getTrackbarPos("Alignement", "Manche")
        alpha=100/cv2.getTrackbarPos("Transparence", "Manche")
        #rotation de l'image texture suivant l'angle du manche et redimensionnement de l'image au dimension de la boite
        #resized_img = cv2.resize(img_manche, (img_width,img_height), interpolation = cv2.INTER_AREA)
        rotated_img=rotate_image(img_manche,angle-90)
        resized_img = cv2.resize(rotated_img, (img_width,img_height), interpolation = cv2.INTER_AREA)
        
        #resized_img = cv2.resize(rotated_img, dim, interpolation = cv2.INTER_AREA)
        #----------------------------------------------------------------------
        #calque_height=abs(y_up_left-y_down_right)
        #calque_width=abs(x_down_right-x_up_left) 
        print("shape", rotated_img.shape)
        dim=resized_img.shape
        #print("w1",abs())
        #print("dim: ", resized_img.shape)
        #alpha=0.4
        Y=abs(y11+y21)//2-offset
        X=x11-fin
        #Y=y21
        # Coller la texture sur le background image/image
        weighted_img = cv2.addWeighted(img[Y:Y+dim[0],X:X+dim[1],:],alpha,resized_img,1-alpha,0)
        # add image to frame
        #print("H ",img_height>=20,"L ",img_width>=300 )
        #print("largeur: ",img_width,"H: ",img_height)
        
       
             
        
        
        img[Y:Y+dim[0] , X:X+dim[1] ] = weighted_img
    #bgrimg = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #cv2.imshow("hsv",bgrimg)
                # Display the resulting frame
    #cv2.imshow('frame2',img)
    return

def texture_color_pixel(hsv,Y,):
   
    return


def accord(ac):
    #(corde, frette)
        switcher={
                "Do":[(2,3),(3,2),(5,1)],
                "Ré":[(4,2),(5,3),(6,2)],
                "Mi":[(2,2),(3,2),(4,1)],
                "Fa":[(1,1),(2,3),(3,3),(4,2),(5,1),(6,1)],
                "Sol":[(1,3),(2,5),(3,5),(4,4),(5,3),(6,3)],
                "La":[(3,2),(4,2),(5,2)],
                "Si":[(2,2),(3,4),(4,4),(5,4),(6,2)],
             }
        return switcher.get(ac,"Accord invalide")





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

    lower_green = np.array([l_h, l_s, l_v])
    upper_green = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_green, upper_green)
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
                
                x22 = int((x+(x+h))/2)
                y22 = int(y+h)
                
                cv2.line(img, (x21, y21), (x22, y22), (0, 0, 255),5)
                #cv2.line(img, (x11, y11), (x21, y21), (0, 0, 255),5) #ligne manche
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
    #   Mauve (x11,y11) : point situé à l'intersection manche-caisse
    #   Bleu  (x21,y21) : point situé au sillet, corde 0 (mi grave)        
    #   Rose  (x22,y22) : point situé au sillet, corde 6 (mi aigu) 
    #   Beige (x12,y12): point point situé à l'intersection manche-caisse en bas

    #   -   -   -                 
    cv2.circle(img, (x11, y11), 5, (255, 0, 122), -1) #point mauve
    cv2.circle(img, (x21, y21), 5, (255, 200, 100), -1) #point bleu clair
    cv2.circle(img, (x22, y22), 5, (120,50, 200), -1) #point rose
    cv2.circle(img, (x12, y12), 5, (120,150, 250), -1) #point beige

    longManche = math.sqrt(math.pow(x21-x11,2)+math.pow(y21-y11,2))
    #print("longueur manche: ", longManche)
    largSillet = math.sqrt(math.pow(x22-x21,2)+math.pow(y22-y21,2))
    #print("largeur Sillet: ", largSillet)

    
              
    
    

    #sur ma guitare, le manche va de la frette 0 à la frette 12 donc le diapason est égal à longManche*2
    if (numeroFretteCaisse == 12):
        #la 12eme frette se trouve à la moitié 
        longDiapason = (longManche*2)
    
    if(numeroFretteCaisse == 14) :
        longDiapason = (longManche-(distance1214*longManche))*2
    

    #matrice de pourcentages pour les distances inter-frettes
    M = matFr()
    for i in range (0,nb):
        print(M[i])


    #on crée les matrices de coordonnées pour les notes
    #en X : on va du sillet à la caisse
    #en Y : du mi grave au mi aigue
    grid = createGrid(int(x11),int(y22),int(x21),int(y21+((y22-y21)/5)), M, longDiapason)
    gridX = createGridX(int(x11),int(y22),int(x21),int(y21+((y22-y21)/5)), M, longDiapason)
    gridY = createGridY(int(x11),int(y22),int(x21),int(y21+((y22-y21)/5)), M, longDiapason)

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
        for y in range(0,6):
            #(xG,yG) = coordonnées dans la grille avant rotation
            xG = gridX[y,x]
            yG = gridY[y,x]
            #(xR,yR) = coordonnées dans la grille après rotation
            xR = gridXR[y,x]
            yR = gridYR[y,x]

            #print("\n-    -   -xG, yG : ",xG, yG,"\n")
            #print("\n-    -   -xR, yR : ",xR, yR,"\n")
            
            #on affiche chaque point de la matrice ayant subi la rotation
            #cv2.circle(img, (int(xR), int(yR)), 2, (0, 255, 0), -1)

    
    
    #La : 3eme corde 7eme case 
    print(grid[:, 3, 6])
    #dessiner une note a la 2eme case, 2eme corde (faire -1 pour l'indice de la corde)
    #dessNote(2,1,gridXR, gridYR, img)
    #dessNote(2,2,gridXR, gridYR, img)



    fa = accord("Fa")
    for (corde,frette) in fa:
        corde -=1
        dessNote(frette,corde,gridXR,gridYR,img)

           
    #print(angleLigne)

    #texture(x11,y11,x12,y12,x21,y21,x22,y22,angleLigne)

    cv2.imshow("Image", img)
    #cv2.imshow("Mask", mask)
    cv2.imshow("Test", test)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()