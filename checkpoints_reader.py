from cv2 import imwrite
from numpy import asarray, zeros, uint8
x=0
w =30
y=20
h=50

p=0
pp=0
image = zeros((4,11,120,160), dtype=uint8)
for i in range(image.shape[0]):
    x +=5
    w += 5
    y += 5
    h += 5

    p = 0
    pp = 0
    for j in range(image.shape[1]):
        if(i==0):
            if(j>0):
                x+=25
                w+=25
                y+=25
                h+=25
                p+=15
                pp+=20
                image[i,j,x+p:w+p, y+pp:h+pp]+=255
                x -= 25
                w -= 25
                y -= 25
                h -= 25
                p -= 15
                pp -= 20
            else:
                x -= 2
                w -= 2
                y -= 2
                h -= 2
                p -= 2
                pp -= 2
                image[i, j, x + p:w + p, y + pp:h + pp] += 255
                x -= 2
                w -= 2
                y -= 2
                h -= 2
                p -= 2
                pp -= 2
        elif(i==1):
            if (j > 0):
                x += 5
                w += 5
                y += 10
                h += 10
                p += 5
                pp += 10
                image[i, j, x + p:w + p, y + pp:h + pp] += 255
                x -= 5
                w -= 5
                y -= 10
                h -= 10
                p -= 5
                pp -= 10
            else:
                x -= 2
                w -= 2
                y -= 2
                h -= 2
                p -= 2
                pp -= 2
                image[i, j, x + p:w + p, y + pp:h + pp] += 255
                x -= 2
                w -= 2
                y -= 2
                h -= 2
                p -= 2
                pp -= 2
        elif (i == 3):
            if (j > 0):
                x -=5
                w -=5
                y -=5
                h -=5
                p -=5
                pp -=5
                image[i, j, x + p:w + p, y + pp:h + pp] += 255
                x +=5
                w +=5
                y += 5
                h += 5
                p +=5
                pp += 5
            else:
                x -= 2
                w -= 2
                y -= 2
                h -= 2
                p -= 2
                pp -= 2
                image[i, j, x + p:w + p, y + pp:h + pp] += 255
                x -= 2
                w -= 2
                y -= 2
                h -= 2
                p -= 2
                pp -= 2
        else:
            if (j > 0):
                x += 5
                w += 5
                y += 5
                h += 5
                p += 5
                pp += 5
                image[i, j, x + p:w + p, y + pp:h + pp] += 255
                x -= 5
                w -= 5
                y -= 5
                h -= 5
                p -= 5
                pp -= 5
            else:

                image[i, j, x + p:w + p, y + pp:h + pp] += 255

for k in range(image.shape[0]):
    for n in range(image.shape[1]):
        imwrite("./images/images/"+"hp_"+str(k)+str(n)+".png", image[k,n,:,:])