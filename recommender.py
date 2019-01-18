import numpy as np
import random
import copy


class Recommender():
    def __init__(self,initRating, initRated, initFeatures, alpha, lbda):
        self.ratingMatrix = np.asarray(initRating)
        self.modifiedRatingMatrix = copy.copy(self.ratingMatrix)
        self.ratedMatrix  = np.asarray(initRated)
        self.m=self.ratingMatrix.shape[0]                                       #m ... Anzahl Filme
        self.n=self.ratingMatrix.shape[1]                                      #n ... Anzahl User
        self.meanM = np.zeros([self.m,1])
        self.numRatingM = np.zeros([self.m,1])                                  
        
        self.features = np.concatenate((np.zeros([self.m,1])+1,np.asarray(initFeatures)),axis = 1)            #1 in erster Spalte als "Nullfeature" berücksichtigt
        self.anzFeatures = self.features.shape[1]
        
        self.alpha = alpha                                                      # learningrate
        self.lbda = lbda                                                        # Überanpassungsparameter
        
        self.param=np.zeros([self.anzFeatures,self.n])+3
        
        self.neighborhoodMatrix = np.diag(np.ones(self.n))
        
        
    def write(self):
        print(" Anzahl Filme: ", self.m, "Anzahl User:", self.n)
        
        for i in range(0, self.m):
            for j in range (0, self.n):
                print('{: 4.2f}'.format(self.ratingMatrix[i][j]), end=' ')
            
            print("|", end= ' ')
            
            for k in range(0,self.anzFeatures):
                print('{: 4.2f}'.format(self.features[i][k]), end = ' ')
            
            print("\n")
            
        print("-----Rated-----")
        
        for i in range(0, self.m):
            for j in range (0, self.n):
                print('{: 1d}'.format(self.ratedMatrix[i][j]), end=' ')
            
            print("\n")
    
    
    def getmeanM(self, nr=-1):
        if nr == -1:
            return self.meanM  
        else:
            return self.meanM[nr]  

            
    def getparam(self, nr=-1):
            if nr == -1:
                return self.param  
            else:
                return self.param[nr]              
    
    
    def getnumRatingM(self, nr=-1):
            if nr == -1:
                return self.numRatingM  
            else:
                return self.numRatingM[nr]  

                
    def addMovie(self, rating, rated, features):
        self.ratingMatrix = np.concatenate((self.ratingMatrix,np.asarray(rating)),axis = 0)
        self.modifiedRatingMatrix = np.concatenate((self.modifiedRatingMatrix,np.asarray(rating)),axis = 0)
        self.m = self.m+ rating.shape[0]
        self.ratedMatrix = np.concatenate((self.ratedMatrix, np.asarray(rated)), axis = 0)
        self.meanM = np.concatenate((self.meanM, np.zeros([rating.shape[0],1])))
        self.numRatingM = np.concatenate((self.numRatingM,np.zeros([rating.shape[0],1]) ))
        
        self.meanMovie((self.m-rating.shape[0]-1, self.m))
        self.features = np.concatenate((self.features, np.concatenate(([[1]],np.asarray(features)), axis=1)) )
        
    
    def addUser(self, rating, rated):
        self.ratingMatrix = np.concatenate((self.ratingMatrix,np.asarray(rating)),axis= 1)
        self.modifiedRatingMatrix = np.concatenate((self.modifiedRatingMatrix,np.asarray(rating)),axis= 1)
        self.n = self.n+ rating.shape[1]
        self.ratedMatrix = np.concatenate((self.ratedMatrix, np.asarray(rated)), axis = 1)  


    def meanMovie(self, crange = (0,0)):                        # numUser lässt Mittelwert und Anzahl der Ratings für einzelnen Film nachberechnen
        
        if crange == (0,0):
            a = 0
            e = self.m
        else:
            a = crange[0]
            e = crange[1]
        
        for j in range(a,e):
            sum =0.0
            anz =0
            for i in range(0,self.n):
                if self.ratedMatrix[j,i] == 1:
                    anz = anz +1
                    sum = sum + self.ratingMatrix[j,i]
            self.meanM[j] = sum/anz
            self.numRatingM[j] = anz
         
         
    def getMeans(self):
        return self.meanM;    
        
        
    def modifyData(self):
        self.meanMovie()
        for i in range(0,self.m):
            for j in range(0,self.n):
                self.modifiedRatingMatrix[i,j] = self.ratingMatrix[i,j]-self.meanM[i]
        #self.write() 

     
    def gradientDescentUpdate(self, deltaMin):
        self.modifyData()
        old = copy.copy(self.param)
        delta = np.zeros([np.shape(old)[0],np.shape(old)[1]])
        
        for j in range(0,self.n):
            sum = 0
            for i in range(0,self.m):
                if self.ratedMatrix[i][j] == 1:
                    sum = sum + (np.inner(self.param[:,j],self.features[i,:]) -self.modifiedRatingMatrix[i][j]) * self.features[i][0]
            self.param[0][j] = self.param[0][j] - self.alpha*sum
            delta[0][j] = abs( self.param[0][j] - old[0][j] )
            
            for k in range(1,self.anzFeatures):
                sum =0
                for i in range(0,self.m):
                    if self.ratedMatrix[i][j]== 1:
                        sum = sum + (np.inner(self.param[:,j],self.features[i,:])-self.modifiedRatingMatrix[i][j])*self.features[i][k]+ self.lbda * self.param[k][j]
                self.param[k][j] = self.param[k][j] - self.alpha*sum
                delta[k][j] = abs(self.param[k][j] - old[k][j])
                          
            self.alpha = 0.99*self.alpha
            
        if delta.max() > deltaMin:
            self.gradientDescentUpdate(deltaMin)
        
        
    #content based prediction
    def predictUserCont(self,user,movie,deltaMin):           
        self.gradientDescentUpdate(deltaMin)
        prediction = np.inner(self.param[:,user],self.features[movie,:]) + self.meanM[movie]
   
        return prediction
     
     
    def calcSimilarity(self, user):
        normU = np.sqrt(np.inner(self.ratingMatrix[:,user],self.ratingMatrix[:,user]))
        for i in range(0,self.n):
            normI = np.sqrt(np.inner(self.ratingMatrix[:,i],self.ratingMatrix[:,i]))
            if i != user:
                self.neighborhoodMatrix[i,user] = np.inner(self.ratingMatrix[:,user], self.ratingMatrix[:,i])/(normU*normI)
                self.neighborhoodMatrix[user,i] = self.neighborhoodMatrix[i,user] 
                
                
                    
    def getNeighborhoodMatrix(self):   
        return self.neighborhoodMatrix
     
     
    #collaborative (neighborhood) prediction ... numNeighbors gibt an wie viele nächste Nachbarn berücksichtigt werden sollen
    def predictUserNeighborhood(self,user,movie, numNeighbors):     
        self.calcSimilarity(user)   
        simvector = self.neighborhoodMatrix[:,user]
        
        k = np.zeros(numNeighbors, dtype = int)
        sum = 0
        div = 0
        shift = 0
        
        for i in range(0,numNeighbors-shift):
            k[i] = int(np.argsort(simvector)[-2-i-shift])
        
            if self.ratedMatrix[movie,k[i]] == 0:

                while shift < self.n-numNeighbors:
                    shift = shift +1
                    k[i] = int(np.argsort(simvector)[-2-i-shift])
                    
                    if self.ratedMatrix[movie,k[i]] == 1:
                        break
                
            sum = sum + simvector[k[i]] * self.ratingMatrix[movie,k[i]]
            div = div + simvector[k[i]]        
                        
        prediction = sum/div    
        return prediction
    
    
    def predictUserMixed(self,user,movie,deltaMin,numNeighbors):
        prediction = (self.predictUserCont(user,movie,deltaMin) + self.predictUserNeighborhood(user,movie,numNeighbors))/2
        return prediction