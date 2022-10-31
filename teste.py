import cv2 as cv
import numpy as np

#camera = cv.VideoCapture("Ambulance3.mp4")

cascade = cv.CascadeClassifier("treinamento/cascade.xml")

while True:
    imagem1 = cv.imread("Ambulance/test/PNMRPUN2EJN2.jpg")
    imagem2= cv.imread("Ambulance/test/IPEKSYZFJD1I.jpg")

    #_, imagemC = camera.read()

    gray = cv.cvtColor(imagem2, cv.COLOR_BGR2GRAY)
    objetos = cascade.detectMultiScale(gray, 1.15, 5)

    for (x, y, w, h) in objetos:
        cv.rectangle(imagem2, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv.imshow("Ambulance", imagem2)

    k = cv.waitKey(1)
    if k == 27:
        break

cv.destroyAllWindows()
#camera.release()

