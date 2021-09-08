  
#Random sample using original mean and covariance vector  
#plotted over contour plot  
	  
N = 100  
  
A =(stats.multivariate_normal.rvs(size=N, mean = mu, cov = sigma))  
          
x = A[:,0]  
y = A[:,1]              
  
plt.plot([mu[0],mu[0]+P[0,0]],[mu[1],mu[1]+P[1,0]],'black')  
plt.plot([mu[0],mu[0]+P[0,1]],[mu[1],mu[1]+P[1,1]],'black')  

 
plt.scatter(x,y)  
plt.contour(X,Y,Z)  
plt.axis('equal')  
plt.title('Contour with P mapped on it and scatter plot')  
plt.show()  
  
#Calculate the  
#sample mean vector and the sample covariance   
#matrix 𝐶 and compare these to 𝝁 and Σ.   
  
#Smu = sample mean   
Smu1 = np.mean(A[:,0])  
Smu2 = np.mean(A[:,1])  
  
Smu = np.array([Smu1, Smu2])  
	  
print('Original mean\n',  
      mu,  
     '\nSample mean\n',  
      Smu)  
#Smu is almost exactly the same as the original vector mean  
  
#C = sample covariance matrix  
C = np.cov(x,y)  
  
print('Original Covarience matrix\n',  
     sigma,  
      '\nCovarience matrix\n',  
      C)  
#C is almost the exact same as the original sigma as well.   
  
#as N is increased the closer the new mean is to the old mean  
# and the closer sigma is to the new covariance matrix   
  
#Diagonalise 𝐶 as 𝐶 = 𝑃𝐷𝑃^-1  
	  
V,P = np.linalg.eigh(C)  
#V = eigenvalues for each vector   
print('eigenvalues\n'  
     , V)  
#p = eigenvector for matrix M   
print('eigenvector\n'  
      , P)  
  
#Matrix D is a matrix made up of  
# the eigenvalues put into a matrix using diagonalization  
# Diagonalization  
# (matrix decomposition)  
D = np.diag(V)  
#D = eigenvalues put into a matrix   
print('D eigenvalues into a matrix\n'  
	      , D)  
  
#PDP^-1    
magic = P@D@np.linalg.inv(P)  
print('magic\n'  
      ,magic)  
  
 
print('C matrix\n',  
	      C,  
	      '\nmagic\n',  
     magic)  
#magic = C   
  
#mean is taken away from A (The random sample)  
plt.figure()  
	  
#subtracting the mean   
A1 = A - Smu  

#P is used as the matrix transformation   
#A1 was transposed for the matrix multiplication.   
Mt = P@A1.T  
	          
x = Mt[0]  
y = Mt[1]              
 
plt.scatter(x,y)  
  
plt.axis("equal")  
plt.show()  
  
newC = np.cov(x,y)  
	  
print('D\n',  
	      D)  
print('newC\n',  
	      newC)  
	  
#D makes up the covariance of X and Y as D makes up   
#the diagonal line from top left to bottom right for the new   
#covariance matrix.   