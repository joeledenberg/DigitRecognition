"""
Created on 09.11.2011

Author: Joel Edenberg
Created for Data Mining crouse project
2011 University of Tartu
"""

import scipy.io as sio
import numpy as np
import pygame.font, pygame.event, pygame.draw
import scipy.optimize as sc
from pygame.locals import *
import heapq

changed = False
counter = 0
Xtrain, Xtest, ytrain, ytest = [], [], [], []
num_hidden = 25
num_input = 900
num_lables = 10
screen = None

def splitData(X, y):
    """Split sample data into training set (80%) and testing set (20%)"""
    
    size1 = X.shape[0] * 0.8
    size2 = X.shape[0] * 0.2
    Xtrain = np.zeros((size1,X.shape[1]))
    Xtest = np.zeros((size2,X.shape[1]))
    ytrain = np.zeros((size1,1))
    ytest = np.zeros((size2,1))
    for i, v in enumerate(np.random.permutation(len(y))):
        #print(i, y[v], len(X[v]))
        
        try:
            Xtrain[i] = X[v]
            ytrain[i] = y[v]
        except:
            Xtest[i-size1] = X[v]
            ytest[i-size1] = y[v]
    return Xtrain, Xtest, ytrain, ytest
    
def calculateImage(background, screen, Theta1, Theta2, lineWidth):
    """Crop and resize the input"""
    
    global changed
    focusSurface = pygame.surfarray.array3d(background)
    focus = abs(1-focusSurface/255)
    focus = np.mean(focus, 2) 
    x = []
    xaxis = np.sum(focus, axis=1)
    for i, v in enumerate(xaxis):
        if v > 0:
            x.append(i)
            break
    for i, v in enumerate(xaxis[ : :-1]):
        if v > 0:
            x.append(len(xaxis)-i)
            break
    
    y = []
    yaxis = np.sum(focus, axis=0)
    for i, v in enumerate(yaxis):
        if v > 0:
            y.append(i)
            break
    for i, v in enumerate(yaxis[ : :-1]):
        if v > 0:
            y.append(len(yaxis)-i)
            break

    try:
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        bound = focus.shape[0]      
        if dx > dy:
            d = dx-dy
            y0t = y[0] - d//2
            y1t = y[1] + d//2+d%2
            if y0t < 0: y0t = y[0]; y1t = y[1] + d
            if y1t > bound: y0t = y[0] - d; y1t = y[1]
            y[0], y[1] = y0t, y1t
        else:
            d = dy-dx
            x0t = x[0] - d//2
            x1t = x[1] + d//2+d%2
            if x0t < 0: x0t = x[0]; x1t = x[1] + d
            if x1t > bound: x0t = x[0] - d; x1t = x[1]
            x[0], x[1] = x0t, x1t 
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        changed = True
        crop_surf =  pygame.Surface((dx,dy))
        crop_surf.blit(background,(0,0),(x[0],y[0],x[1],y[1]), special_flags=BLEND_RGBA_MAX)
        scaledBackground = pygame.transform.smoothscale(crop_surf, (30, 30))
            
        image = pygame.surfarray.array3d(scaledBackground)
        image = abs(1-image/253)
        image = np.mean(image, 2) 
        image = np.matrix(image.ravel())
        drawPixelated(image, screen)
        (value, prob), (value2, prob2) = probabilty(Theta1,Theta2,image)
        prob = round(prob,1)
        prob2 = round(prob2, 1)
                   
        myLabel = showStats(lineWidth, value, prob)
        myLabelSmall = showStatsSmall(lineWidth, value2, prob2)
        (x,y) = screen.get_size()
        screen.blit(myLabel, (17, y-90))
        screen.blit(myLabelSmall, (20, y-38))
    except:
        image = np.zeros((30,30))

    return image
    
def sigmoid(z):
    """Calculate sigmoid function"""
    
    return 1/(1+np.power(np.e,-z))

def probabilty(Theta1, Theta2, X):
    """Classify number and caluclate probability"""
    
    X = np.append(np.ones(shape=(X.shape[0],1)),X,axis=1)
    input = Theta1*np.matrix(X.transpose())
    hiddenLayer = sigmoid(input)
    hiddenLayer = np.append(np.ones(shape=(1,hiddenLayer.shape[1])),hiddenLayer,axis=0)
    proba = sigmoid(Theta2*hiddenLayer)
    l0 = np.array(proba.ravel())[0]
    l1 = heapq.nlargest(2, l0)
    proba2 = l1[1]
    estimate2 = int(np.where(l0==l1[1])[0]+1)
    estimate2 = estimate2 if estimate2<10 else 0
    number = int(proba.argmax(0).transpose())
    estimate = number+1 if number<9 else 0

    return (estimate, float(proba[number])*100), (estimate2, proba2*100)


def probabiltyForDrawing(Theta1, Theta2, X):
    """Calculate the prdeiction probabilties"""
    X = np.append(np.ones(shape=(X.shape[0],1)),X,axis=1)
    input = Theta1*np.matrix(X.transpose())
    hiddenLayer = sigmoid(input)
    hiddenLayer = np.append(np.ones(shape=(1,hiddenLayer.shape[1])),hiddenLayer,axis=0)
    proba = sigmoid(Theta2*hiddenLayer)
    numbers = proba.argmax(0).transpose()+1
    return numbers

def returnAccuracy(prob, y):
    """Calculate the prediction accuracy"""
    count = 0
    for i in range(len(prob)):
        if int(prob[i]) == int(y[i]):
            count += 1
        elif int(prob[i]) == 10 and int(y[i]) == 0:
            count += 1
    return round((count/len(prob))*100,2)

def get_key():
    """Get key event"""
    while 1:
        event = pygame.event.poll()
        if event.type == KEYDOWN:
            return event.key
        else:
            pass

def display_box(screen, message):
    "Print a message in a box on the screen"
    
    fontobject = pygame.font.Font(None,120)
    pygame.draw.rect(screen, (0,0,0),
                   ((screen.get_width() / 2) - 100,
                    (screen.get_height()) - 170,
                    70,90), 0)
    if len(message) != 0:
        screen.blit(fontobject.render(message, 1, (255,255,255)),
                ((screen.get_width() / 2) - 110, (screen.get_height()) -168))
        pygame.display.flip()


def ask(screen, question):
    """create input box for entering correct value of y"""
    
    pygame.font.init()
    current_string = str()
    display_box(screen, question + " " + current_string+"")
    while 1:
        inkey = get_key()
        if inkey == K_BACKSPACE:
            current_string = current_string[0:-1]
        elif inkey == K_RETURN:
            break
        elif inkey == K_MINUS:
            current_string.append("_")
        elif inkey <= 127:
            current_string+= (chr(inkey))
        display_box(screen, question + " " + current_string+"")
    return current_string
    
def checkKeys(myData):
    """test for various keyboard inputs"""
    
    (event, background, drawColor, lineWidth, keepGoing, screen, image) = myData
    
    if event.key == pygame.K_q:
        keepGoing = False
    elif event.key == pygame.K_c:
        background.fill((255, 255, 255))
        drawPixelated(np.zeros((30,30)), screen)
    elif event.key == pygame.K_s:
        global changed
        try:
            mat_contents = sio.loadmat('newX.mat')
            X = mat_contents['X']
            X = np.append(X,image, axis=0)
        except:
            X = image
        answer = np.matrix(int(ask(screen, "")))
        try:
            mat_contents = sio.loadmat('newy.mat')
            y = mat_contents['y']
            y = np.append(y,answer, axis=0)
        except:
            y = answer
        if changed:
            sio.savemat('newX.mat', {'X': X})
            sio.savemat('newy.mat', {'y': y})
            changed = False

        background.fill((255, 255, 255))
        drawPixelated(np.zeros((30,30)), screen)

    elif event.key == pygame.K_t:
        screen.fill((255, 255, 255))
        myFont1 = pygame.font.SysFont("Verdana", 55)
        myFont2 = pygame.font.SysFont("Verdana", 17)
        myFont3 = pygame.font.SysFont("Verdana", 9)
        screen.blit(myFont1.render("Please wait!", 1, ((0, 0, 0))), (365, 90))
        screen.blit(myFont2.render("Neural network training in progress...", 1, ((50, 50, 50))), (368, 150))
        screen.blit(myFont3.render("Depending on the training data size this could take long time", 1, ((80, 80, 80))), (370, 190))
        pygame.display.flip()
        global Xtrain; global Xtest; global ytrain; global ytest
        mat_contents = sio.loadmat('newX.mat')
        Xs = mat_contents['X']
        mat_contents = sio.loadmat('newy.mat')
        ys = mat_contents['y']
        Xtrain, Xtest, ytrain, ytest = splitData(Xs,ys)
        
        rndInit = randomInitialization(25*901+10*26)
        answer =  sc.fmin_cg(calculateJ, rndInit, calculateGrad, maxiter=100,  disp=True, callback=callback)
        Theta1 = np.reshape(answer[:num_hidden*(num_input+1)], (num_hidden,-1))
        Theta2 = np.reshape(answer[num_hidden*(num_input+1):], (num_lables,-1))

        acc = returnAccuracy(probabiltyForDrawing(Theta1, Theta2, Xtest), ytest)
        sio.savemat('scaledTheta.mat', {'t': answer, 'acc': acc})
        screen.fill((0, 0, 0))
        background.fill((255, 255, 255))
    elif event.key == pygame.K_v:
        drawStatistics(screen)

    myData = (event, background, drawColor, lineWidth, keepGoing)
    return myData


def showStats(lineWidth, value, prob):
    """ shows the current statistics """
    
    myFont = pygame.font.SysFont("Verdana", 50)
    stats = "Estimate: %s    P: %s" % (value, prob)
    statSurf = myFont.render(stats+"%", 1, ((255, 255, 255)))
    return statSurf


def showStatsSmall(lineWidth, value, prob):
    """ shows the current statistics """
    
    myFont = pygame.font.SysFont("Verdana", 25)
    stats = "Second estimate: %s (%s" % (value, prob)
    statSurf = myFont.render(stats+"%)", 1, ((235, 235, 235)))
    return statSurf

def drawPixelated(A, screen):  
    """Draw 30x30 image of input""" 
    
    A = A.ravel()
    A = (255-A*255).transpose()
    size = 30
    for x in range(size):
        for y in range(size):
            z=x*30+y
            c = int(A[z])
            pygame.draw.rect(screen,(c,c,c),(x*11+385,15+y*11,11,11))



def drawStatistics(screen):  
    """Draw statistics about training set"""

    mat_contents = sio.loadmat('newX.mat')
    Xs = mat_contents['X']
    mat_contents = sio.loadmat('newy.mat')
    ys = mat_contents['y']
    Xtrain, Xtest, ytrain, ytest = splitData(Xs,ys)
    mat_contents = sio.loadmat('scaledTheta.mat')
    acc = float(mat_contents['acc'])
    y = ys.ravel().tolist()

    myFont = pygame.font.SysFont("Verdana", 24)
    myFont2 = pygame.font.SysFont("Verdana", 18)
    myFont3 = pygame.font.SysFont("Verdana", 16)
    pygame.draw.rect(screen,(255,255,255),(370,0,730,360))
    screen.blit(myFont.render("Samples: %d" % (Xs.shape[0]), 1, ((0, 0, 0))), (400, 30))
    screen.blit(myFont.render("Accuracy: %s" % str(acc)+"%", 1, ((0, 0, 0))), (400, 60))
    screen.blit(myFont3.render("SAMPLE DISTRIBUTION:", 1, ((0, 0, 0))), (400, 100))
    screen.blit(myFont2.render("Count 0 = %s" % (y.count(0)), 1, ((0, 0, 0))), (400, 120))
    for i in range(9):
        screen.blit(myFont2.render("Count %s = %s" % (i+1, y.count(i+1)), 1, ((0, 0, 0))), (400, 140+i*20))


def randomInitialization(i, epsilon=0.12):
    """For symmetry breaking initialize random valued thetas"""
    
    return np.random.rand(i,1)*2*epsilon-epsilon


def sigmoidGradient(z):
    """Gradient of sigmoid function"""
    
    return np.multiply(sigmoid(z),(1-sigmoid(z)));


def vectorize(v1, v2):
    """Merge to vectors together"""
    
    return np.append(np.ravel(v1), np.ravel(v2))


def backProp(p, num_input, num_hidden, num_labels, X, yvalue, l=0.2):
    """Backpropagation algorithm of neural network"""
    
    Theta1 = np.reshape(p[:num_hidden*(num_input+1)], (num_hidden,-1))
    Theta2 = np.reshape(p[num_hidden*(num_input+1):], (num_labels,-1))
    m = len(X)
    delta1 = 0
    delta2 = 0
    for t in range(m):

        a1 = np.matrix(np.append([1],X[t],axis=1)).transpose()
        z2 = Theta1*a1
        a2 = np.append(np.ones(shape=(1,z2.shape[1])), sigmoid(z2),axis=0)
        z3 = Theta2*a2
        a3 = sigmoid(z3)
        w = np.zeros((num_labels,1))
        w[int(yvalue[t])-1] = 1
        d3 = (a3-w)
        d2 = np.multiply(Theta2[:,1:].transpose()*d3, sigmoidGradient(z2))
        delta1 += d2*a1.transpose()
        delta2 += d3*a2.transpose()
        
    
    Theta1_grad = (1/m)*delta1 + (l/m)*np.append(np.zeros(shape=(Theta1.shape[0],1)), Theta1[:,1:], axis=1);
    Theta2_grad = (1/m)*delta2 + (l/m)*np.append(np.zeros(shape=(Theta2.shape[0],1)), Theta2[:,1:], axis=1);
    answer = vectorize(Theta1_grad, Theta2_grad)
    return answer


def J(theta, num_input, num_hidden, num_lables,X, yvalue, l=0.2):
    """Cost funtion"""
    
    Theta1 = np.reshape(theta[:num_hidden*(num_input+1)], (num_hidden,-1))
    Theta2 = np.reshape(theta[num_hidden*(num_input+1):], (num_lables,-1))
    m = len(X)
    X = np.append(np.ones(shape=(X.shape[0],1)),X,axis=1)
    J = 0
    for i in range(m):
        x = np.matrix(X[i])
        w = np.zeros((10,1))
        w[int(yvalue[i])-1] = 1
        hx = sigmoid(Theta2*np.append([[1]], sigmoid(Theta1*x.transpose()), axis=0))
        J += sum(-w.transpose()*np.log(hx)-(1-w).transpose()*np.log(1-hx))
    J = J/m
    J += (l/(2*m))*(sum(sum(Theta1[:,1:]**2)) + sum(sum(Theta2[:,1:]**2)))    
    return float(J)

def calculateGrad(p):
    """Backpropagation method wrapper for optimization function"""
    
    return backProp(p, 900, 25, 10, Xtrain,ytrain)
    
def calculateJ(p):
    """Costfuntion wrapper for optimization function"""
    
    return J(p, 900, 25, 10, Xtrain, ytrain)

def callback(p):
    """Update GUI as neural network is being trained"""
    
    global counter
    global screen
    pygame.event.get()
    counter += 1
    myFontDots = pygame.font.SysFont("Verdana", 110)
    dots= []
    if counter >= 9:
        counter = 1
        pygame.draw.rect(screen,(255,255,255),(355,220,400,250))
        dots = []
    for i in range(counter):
        dots.append(".")
    screen.blit(myFontDots.render("".join(dots), 1, ((150, 150, 150))), (355, 140))
    pygame.display.flip()

    
def main():
    """Main method. Draw interface"""
    
    global screen
    pygame.init()
    screen = pygame.display.set_mode((730, 450))
    pygame.display.set_caption("Handwriting recognition")
    
    background = pygame.Surface((360,360))
    background.fill((255, 255, 255))
    background2 = pygame.Surface((360,360))
    background2.fill((255, 255, 255))
    
    clock = pygame.time.Clock()
    keepGoing = True
    lineStart = (0, 0)
    drawColor = (255, 0, 0)
    lineWidth = 15
    
    inputTheta = sio.loadmat('scaledTheta.mat')
    theta = inputTheta['t']
    num_hidden = 25
    num_input = 900
    num_lables = 10

    Theta1 = np.reshape(theta[:num_hidden*(num_input+1)], (num_hidden,-1))
    Theta2 = np.reshape(theta[num_hidden*(num_input+1):], (num_lables,-1))

    pygame.display.update()
    image = None
            
    while keepGoing:
        
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                keepGoing = False
            elif event.type == pygame.MOUSEMOTION:
                lineEnd = pygame.mouse.get_pos()
                if pygame.mouse.get_pressed() == (1, 0, 0):
                    pygame.draw.line(background, drawColor, lineStart, lineEnd, lineWidth)
                lineStart = lineEnd
            elif event.type == pygame.MOUSEBUTTONUP:
                screen.fill((0, 0, 0))
                screen.blit(background2, (370, 0))
                #w = threading.Thread(name='worker', target=worker)
                image = calculateImage(background, screen, Theta1, Theta2, lineWidth)

            elif event.type == pygame.KEYDOWN:
                myData = (event, background, drawColor, lineWidth, keepGoing, screen, image)
                myData = checkKeys(myData)
                (event, background, drawColor, lineWidth, keepGoing) = myData
        
        
        screen.blit(background, (0, 0))
        pygame.display.flip()

        
if __name__ == "__main__":
    main()