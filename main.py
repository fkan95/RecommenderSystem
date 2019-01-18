import recommender as rec
import numpy as np     
     
     
     
#Vorhersagen für:     
user = 3
movie1 = 0
movie2 = 6 


#Systemparameter
nearestNeighbors = 2     
     
alpha = 0.1                              #learningrate 
lbda = 0.1                               #Überanpassungsparameter
deltaMin = 0.001                         #Genauigkeitsschwelle                                                        


rating=np.array([[5,3,1.5,0,3.5],
                 [4,2.5,0,5,5],
                 [5,0,1,4,3.5],
                 [0,4.5,5,1,2],
                 [0,0,1,4,0],
                 [1,0,0,0.5,0.5],
                 [0,4,1.5,0,0],
                 [3,0,3,1,0],
                 [4,3.5,0,3,3.5]])

rated=np.array([[1,1,1,0,1],
                [1,1,1,1,1],
                [1,0,1,1,1],
                [0,1,1,1,1],
                [0,0,1,1,0],
                [1,0,0,1,1],
                [1,1,1,0,0],
                [1,0,1,1,1],
                [1,1,0,1,1]])

features = np.array([[7,2],
                     [10,0],
                     [8,2], 
                     [1,10], 
                     [8,1],
                     [0,10],
                     [3,9],
                     [1,6], 
                     [6,7]])/10


neuerFilm = np.array([[3,3,2.5,4,5]])
isRated = np.array([[1,1,1,1,1]])
neuesFeature = np.array([[8,6]])/10
'''
neuerUser = np.array([[9,8,7,6,5,4]])
neuesUserRating = np.array([[1,1,1,1,1,1]])
'''


A = rec.Recommender(rating, rated, features, alpha, lbda)


print("-------\n\n")

A.addMovie(neuerFilm, isRated, neuesFeature)
A.write()

A.meanMovie()

A.calcSimilarity(3)
#print("Nachbarschaftsmatrix: ")
#print(A.getNeighborhoodMatrix())


print("-------\n\n")



print("\n\nVorhersage der Bewertung von Benutzer ", user, "für Film ", movie1, ": ", A.predictUserCont(user,movie1,deltaMin)[0], " Sterne. (Content)")
print("\n\nVorhersage der Bewertung von Benutzer ", user, "für Film ", movie2, ": ", A.predictUserCont(user,movie2,deltaMin)[0], " Sterne. (Content)")
input()

print("\n\nVorhersage der Bewertung von Benutzer ", user, "für Film ", movie1, ": ", A.predictUserNeighborhood(user,movie1, nearestNeighbors), " Sterne. (Neighborhood)")
print("\n\nVorhersage der Bewertung von Benutzer ", user, "für Film ", movie2, ": ", A.predictUserNeighborhood(user,movie2, nearestNeighbors), " Sterne. (Neighborhood)")
input()

print("\n\nVorhersage der Bewertung von Benutzer ", user, "für Film ", movie1, ": ", A.predictUserMixed(user,movie1,deltaMin,nearestNeighbors)[0], " Sterne. (Mixed)")
print("\n\nVorhersage der Bewertung von Benutzer ", user, "für Film ", movie2, ": ", A.predictUserMixed(user,movie2,deltaMin,nearestNeighbors)[0], " Sterne. (Mixed)")
input()