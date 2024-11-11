import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Design Variables
#Number of joints n>7
#length of elements li
#Cross sectional area xi

#Design Parameters
#High strength Steel (600 Mpa yield strength) with young's mod E = 200Gpa and density of rho = 8000 kg/m^3
#Square Cross section Ai = xi^2
#eff l = li

#boundry conditions 
#joint a is a hinge, joint B is a roller

#loads
# p = 1.5*66248 N 

#Design Constraints 
#FOS and Buckling: 4
#Slenderness ratio: should be <= 500 in tensino and compression
#Strike Resistance: Must be stable if and arbitrary truss element is removed

#Geometric Constraints 
# Min joint distance is 2m
# 5m <= Lab <= 10m, 0m <= Lfg <= 5m
# lcd = lde = 5m, lbc = lef = 30m 

#lengths
lab = 10
    # lad is not needed with how the code is defined currently 
lcg = 25.0
lbc = 30.0 #m
lef = 30.0 #m
lde = 5.0 #m
lcd = 5.0 #m
lfg = 5 #m

nodeCords=np.array([[0.0,0.0],[lab,0],[lab,lbc],[lab-lcd,lbc],[lab-lcd,lbc+lde],[lab+lcg,lbc],[lab+lcg,lbc+lfg]])
print(nodeCords)

#Nodal Coordinates
elemNodes=np.array([[0,1],[1,2],[0,3],[3,4],[2,5],[5,6],[4,6],[2,3],[1,3],[0,2],[3,6],[4,2]]) #Element connectivity: near node and far node
modE=np.array([[200e9],[200e9],[200e9],[200e9],[200e9],[200e9],[200e9],[200e9],[200e9],[200e9],[200e9],[200e9]]) #Young's modulus
Area=np.array([[250],[250],[250],[250],[250],[250],[250],[250],[250],[250],[250],[250]]) #Cross section area
DispCon=np.array([[0,1,0.0],[0,2,0.0],[1,2,0.0]]) #Displacement constraints
Fval=np.array([[5,2,-1.0]]) #Applied force = 1.5*66248


scale=.1 # Scale factor for viewing deformation plots only


#Problem Initialization
nELEM=elemNodes.shape[0] # Number of elements
nNODE=nodeCords.shape[0] # Number of nodes
nDC=DispCon.shape[0] # Number og constrained DOF
nFval=Fval.shape[0] # Number of DOFs where forces are applied
NDOF=nNODE*2 # Total number of DOFs
uDisp=np.zeros((NDOF,1)) #Displacement vector is NDOF X 1
forces=np.zeros((NDOF,1)) # Forces vector is NDOF X 1
Stiffness=np.zeros((NDOF,NDOF)) # Stiffness matrix is NDOF X NDOF
Stress=np.zeros((nELEM)) # Stress is nELEM X 1 vector
kdof=np.zeros((nDC)) # All known DOFs nDC x 1 vector
xx=nodeCords[:,0] # All X-coordinates of the nodes nNODE X 1 vector
yy=nodeCords[:,1] # All Y-coordinates of the nodes nNODE X 1 vector

L_elem=np.zeros((nELEM)) # All lengths of trusses nELEM X 1 vector

# cost analysis
# C = 0 
# for i in range(len(L_elem)):
#     C += Area[i]*L_elem[i]
# C = C*(1+nNODE)
# print(C)

#Building the displacement array
for i in range(nDC): #looping over the number of known degrees of freedom
    indice=DispCon[i,:]
    v=indice[2] #value of the known displacement 
    v=v.astype(float)
    indice=indice.astype(int)
    kdof[i]=indice[0]*2+indice[1]-1 # The corresponding degree of freedom that is constrained is assigned to kdof[i]
    # print(indice)
    # print(kdof)
    uDisp[indice[0]*2+indice[1]-1]=v # The corresponding displacement value is assigned to uDisp
    
#Building the force array

for i in range(nFval): #looping over the dofs where forces are applied
    indice2=Fval[i,:]
    v=indice2[2]
    v=v.astype(float)
    indice2=indice2.astype(int)
    forces[indice2[0]*2+indice2[1]-1]=v # Assigning the value of the force in the forces vector



#Identifying known and unknown displacement degree of freedom
kdof=kdof.astype(int) #Contains all degrees of freedom with known displacement
ukdof=np.setdiff1d(np.arange(NDOF),kdof) #Contains all degrees of freedom with unknown displacement


#Loop over all the elements

for e in range(nELEM):
    indiceE=elemNodes[e,:] #Extracting the near and far node for element 'e'
    Y=modE[e]
    Ae=Area[e]
    elemDOF=np.array([indiceE[0]*2,indiceE[0]*2+1,indiceE[1]*2,indiceE[1]*2+1]) #Contains all degrees of freedom for element 'e'
    elemDOF=elemDOF.astype(int)
    xa=xx[indiceE[1]]-xx[indiceE[0]]
    ya=yy[indiceE[1]]-yy[indiceE[0]]
    len_elem=np.sqrt(xa*xa+ya*ya) #length of the element 'e'
    c=xa/len_elem #lambda x
    s=ya/len_elem #lambda y


    
    # Step 1. Define elemental stiffness matrix
    ke = (Y*Ae/len_elem)*np.array([[1,-1],[-1,1]])
    
    # Step 2. Transform elemental stiffness matrix from local to global coordinate system
    T = np.array([[c,s,0,0],[0,0,c,s]])
    k2 = np.matmul(T.transpose(), np.matmul(ke,T))
    
    
    # Step 3. Assemble elemental stiffness matrices into a global stiffness matrix
    Stiffness[np.ix_(elemDOF,elemDOF)] += k2


# Step 4. Partition the stiffness matrix into known and unknown dofs
kuu = Stiffness[np.ix_(ukdof,ukdof)]
kuk = Stiffness[np.ix_(ukdof,kdof)]
kku = kuk.transpose()
kkk = Stiffness[np.ix_(kdof,kdof)]


# Step 4a. Solve for the unknown dofs and reaction forces
f_known = forces[ukdof]-np.matmul(kuk,uDisp[kdof])
temp = np.linalg.inv(kuu)
uDisp[np.ix_(ukdof)] = np.linalg.solve(kuu,f_known)

forces[np.ix_(kdof)]= np.matmul(kku,uDisp[np.ix_(ukdof)])+ np.matmul(kkk,uDisp[np.ix_(kdof)])
plt.figure(300)

# Step 5. Evaluating Internal Forces and stresses
for e in range(nELEM):
    indiceE=elemNodes[e,:]
    Y=modE[e]
    Ae=Area[e]
    elemDOF=np.array([indiceE[0]*2,indiceE[0]*2+1,indiceE[1]*2,indiceE[1]*2+1])
    elemDOF=elemDOF.astype(int)
    xa=xx[indiceE[1]]-xx[indiceE[0]]
    ya=yy[indiceE[1]]-yy[indiceE[0]]
    len_elem=np.sqrt(xa*xa+ya*ya)
    L_elem[e]=len_elem
    c=xa/len_elem
    s=ya/len_elem
    
    #Elemental Stiffness Matrix
    ke - (Y*Ae/len_elem)*np.array([[1,-1],[-1,1]])
    # Transformation Matrix
    T = np.array([[c,s,0,0],[0,0,c,s]])
    
    #Internal forces
    Fint = np.matmul(ke,T@uDisp[np.ix_(elemDOF)])
    #Stress
    Stress[e] = Fint[1]/Ae
    
    
    
    
  
    plt.plot(np.array([xx[indiceE[0]],xx[indiceE[1]]]),np.array([yy[indiceE[0]],yy[indiceE[1]]]))
    plt.plot(np.array([xx[indiceE[0]]+uDisp[indiceE[0]*2]*scale,xx[indiceE[1]]+uDisp[indiceE[1]*2]*scale]),np.array([yy[indiceE[0]]+uDisp[indiceE[0]*2+1]*scale,yy[indiceE[1]]+uDisp[indiceE[1]*2+1]*scale]),'--')


plt.xlim(min(xx)-abs(max(xx)/10), max(xx)+abs(max(xx)/10))
plt.ylim(min(yy)-abs(max(yy)/10), max(yy)+abs(max(xx)/10))
plt.gca().set_aspect('equal', adjustable='box')
pduDisp = pd.DataFrame({'disp': uDisp[:,0]})
pdforces=pd.DataFrame({'forces': forces[:,0]})
pdStress=pd.DataFrame({'Stress': Stress})
pdLen=pd.DataFrame({'Length': L_elem})
#Displaying the results
print(pduDisp)
print(pdforces)
print(pdStress)
 

plt.show()

