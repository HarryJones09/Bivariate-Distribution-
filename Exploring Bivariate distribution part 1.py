import numpy as np   
import matplotlib.pyplot as plt  
import scipy.stats as stats  
import seaborn as sns  
sns.set_theme()  
  
  
#mean  
mu = np.array([-2,    
              -4])    
#covariance matrix  
sigma = np.array([[3,-1],    
	              [-1,9]])    


# wireframe plot  
x = np.linspace(-10,10,101)  
y = np.linspace(-10,10,101)  
	  
X, Y = np.meshgrid(x,y)  
Z = np.zeros(X.shape)  
  
	  
#Traverse through the len of y and x   
for i in range(len(y)):  
    for j in range(len(x)):  
       point= np.array([X[i,j],Y[i,j]])  
       Z[i,j] =(stats.multivariate_normal.pdf(point, mu, sigma))  
	          
              
fig = plt.figure()  
ax = fig.add_subplot(111, projection='3d')  
  
ax.plot_wireframe(X,Y,Z)  
 
plt.xlabel('x ')  
plt.ylabel('y')  
plt.show()  
  
  
# Contour plot   
plt.figure()  
plt.contour(X,Y,Z)  

plt.axis('equal')  
plt.show()  
  
  
V,P = np.linalg.eigh(sigma)  
#V = eigenvalues for each vector   
print('eigenvalues\n', V)  
#p = eigenvector for matrix M   
print('eigenvector\n', P)  
	  
#Matrix D is a matrix made up of  
# the eigenvalues put into a matrix using diagonalisation  
# Diagonalisation  
# (matrix decomposition)  
D = np.diag(V)  
#D = eigenvalues put into a matrix   
print('D eigenvalues into a matrix\n', D)  
  
#PDP^-1    
magic = P@D@np.linalg.inv(P)  
print('magic\n'  
      ,magic)  
	  
#contour plot with P mapped onto it   
  
plt.figure()  

  
plt.plot([mu[0],mu[0]+P[0,0]],[mu[1],mu[1]+P[1,0]],'black')  
plt.plot([mu[0],mu[0]+P[0,1]],[mu[1],mu[1]+P[1,1]],'black')  
  
  
plt.title('Contour with P mapped on it')  
plt.xlabel('x ')  
plt.ylabel('y')  
	  
plt.axis('equal')  
plt.contour(X,Y,Z)  
  
  
plt.show()  