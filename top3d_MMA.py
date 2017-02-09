from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
import nlopt
import time


#Default input parameters
class para:
    nelx = 30
    nely = 10
    nelz = 5
    volfrac = 0.5
    rmin = 1.2
    penal = 3.0
    ft = 0

class eMat:
    def __init__(self,nelx,nely,nelz):
        self.nelx=nelx
        self.nely=nely
        self.nelz=nelz
    def Mat(self):
        nelx=self.nelx
        nely=self.nely
        nelz=self.nelz
        Mat = np.zeros((nelx * nely * nelz, 24), dtype=int)
        for elz in range(nelz):
            for elx in range(nelx):
                for ely in range(nely):
                    el = ely + (elx * nely) + elz * (nelx * nely)
                    n1 = elz * (nelx + 1) * (nely + 1) + (nely + 1) * elx + ely
                    n2 = elz * (nelx + 1) * (nely + 1) + (nely + 1) * (elx + 1) + ely
                    n3 = (elz + 1) * (nelx + 1) * (nely + 1) + (nely + 1) * elx + ely
                    n4 = (elz + 1) * (nelx + 1) * (nely + 1) + (nely + 1) * (elx + 1) + ely
                    Mat[el, :] = np.array(
                        [3 * n1 + 3, 3 * n1 + 4, 3 * n1 + 5, 3 * n2 + 3, 3 * n2 + 4, 3 * n2 + 5, \
                         3 * n2, 3 * n2 + 1, 3 * n2 + 2, 3 * n1, 3 * n1 + 1, 3 * n1 + 2, \
                         3 * n3 + 3, 3 * n3 + 4, 3 * n3 + 5, 3 * n4 + 3, 3 * n4 + 4, 3 * n4 + 5, \
                         3 * n4, 3 * n4 + 1, 3 * n4 + 2, 3 * n3, 3 * n3 + 1, 3 * n3 + 2])
        return Mat

class FE:
    def __init__(self,x,nelx,nely,nelz,volfrac,rmin,penal,ft,Emin,Emax,KE):
        self.x=x
        self.nelx=nelx
        self.nely=nely
        self.nelz=nelz
        self.volfrac=volfrac
        self.rmin=rmin
        self.penal=penal
        self.ft=ft
        self.Emin=Emin
        self.Emax=Emax
        self.KE=KE
    def Usolution(self):
        x=self.x
        nelx=self.nelx
        nely=self.nely
        nelz=self.nelz
        volfrac=self.volfrac
        rmin=self.rmin
        penal=self.penal
        ft=self.ft
        Emin=self.Emin
        Emax=self.Emax
        KE=self.KE
        # dofs:
        ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
        # FE: Build the index vectors for the for coo matrix format.
        ee = eMat(nelx,nely,nelz)
        edofMat = ee.Mat()
        # Construct the index pointers for the coo format
        iK = np.kron(edofMat, np.ones((24, 1))).flatten()
        jK = np.kron(edofMat, np.ones((1, 24))).flatten()
        # USER - DEFINED LOAD DOFs
        kl = np.arange(nelz + 1)
        loadnid = kl * (nelx + 1) * (nely + 1) + (nely + 1) * (nelx + 1) - 1  # Node IDs
        loaddof = 3 * loadnid + 1  # DOFs
        # USER - DEFINED SUPPORT FIXED DOFs
        [jf, kf] = np.meshgrid(np.arange(nely + 1), np.arange(nelz + 1))  # Coordinates
        fixednid = (kf) * (nely + 1) * (nelx + 1) + jf  # Node IDs
        fixeddof = np.array([3 * fixednid, 3 * fixednid + 1, 3 * fixednid + 2]).flatten()  # DOFs
        # BC's and support
        dofs = np.arange(3 * (nelx + 1) * (nely + 1) * (nelz + 1))
        free = np.setdiff1d(dofs, fixeddof)
        # Solution and RHS vectors
        f = np.zeros((ndof, 1))
        u = np.zeros((ndof, 1))
        # Set load
        f[loaddof, 0] = -1
        # Setup and solve FE problem
        sK = ((KE.flatten()[np.newaxis]).T * (Emin + x ** penal * (Emax - Emin))).flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        # Remove constrained dofs from matrix
        K = K[free, :][:, free]
        # Solve system
        u[free, 0] = spsolve(K, f[free, 0])
        return u

# Filter: Build (and assemble) the index+data vectors for the coo matrix format
class FILTERMATRIX:
    def __init__(self,nelx,nely,nelz,rmin):
        self.nelx=nelx
        self.nely=nely
        self.nelz=nelz
        self.rmin=rmin
    def assembly(self):
        nelx=self.nelx
        nely=self.nely
        nelz=self.nelz
        rmin=self.rmin
        nfilter = nelx * nely * nelz * ((2 * (np.ceil(rmin) - 1) + 1) ** 3)
        iH = np.zeros(nfilter)
        jH = np.zeros(nfilter)
        sH = np.zeros(nfilter)
        cc = 0
        for z in range(nelz):
            for i in range(nelx):
                for j in range(nely):
                    row = i * nely + j + z * (nelx * nely)
                    kk1 = int(np.maximum(i - (np.ceil(rmin) - 1), 0))
                    kk2 = int(np.minimum(i + np.ceil(rmin), nelx))
                    ll1 = int(np.maximum(j - (np.ceil(rmin) - 1), 0))
                    ll2 = int(np.minimum(j + np.ceil(rmin), nely))
                    mm1 = int(np.maximum(z - (np.ceil(rmin) - 1), 0))
                    mm2 = int(np.minimum(z + np.ceil(rmin), nelz))
                    for m in range(mm1, mm2):
                        for k in range(kk1, kk2):
                            for l in range(ll1, ll2):
                                col = k * nely + l + m * (nelx * nely)
                                fac = rmin - np.sqrt((i - k) * (i - k) + (j - l) * (j - l) + (z - m) * (z - m))
                                iH[cc] = row
                                jH[cc] = col
                                sH[cc] = np.maximum(0.0, fac)
                                cc = cc + 1
        # Finalize assembly and convert to csc format
        H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely * nelz, nelx * nely * nelz)).tocsc()
        return H


#Optimization Problem Definition
def objfunc(x,grad):
    if grad.size > 0:
        # Default input parameters
        p = para()
        nelx = p.nelx
        nely = p.nely
        nelz = p.nelz
        volfrac = p.volfrac
        rmin = p.rmin
        penal = p.penal
        ft = p.ft  # ft==0 -> sens, ft==1 -> dens
        # Max and min stiffness
        Emin = 1e-3
        Emax = 1.0
        # list to array
        # Finalize assembly and convert to csc format
        HH = FILTERMATRIX(nelx, nely, nelz, rmin)
        H = HH.assembly()
        Hs = H.sum(1)
        # KE Matrix
        KE = lk()
        # FE
        uu = FE(x, nelx, nely, nelz, volfrac, rmin, penal, ft, Emin, Emax, KE)
        u = uu.Usolution()
        # sensitivity
        dv = np.ones(nely * nelx * nelz)
        dc = np.ones(nely * nelx * nelz)
        ce = np.ones(nely * nelx * nelz)
        ee = eMat(nelx, nely, nelz)
        edofMat = ee.Mat()
        ce[:] = (
        np.dot(u[edofMat].reshape(nelx * nely * nelz, 24), KE) * u[edofMat].reshape(nelx * nely * nelz, 24)).sum(1)
        dc[:] = (-penal * x ** (penal - 1) * (Emax - Emin)) * ce
        dv[:] = np.ones(nely * nelx * nelz)
        # Sensitivity filtering:
        if ft == 0:
            dc[:] = np.asarray((H * (x * dc))[np.newaxis].T / Hs)[:, 0] / np.maximum(0.001, x)
        elif ft == 1:
            dc[:] = np.asarray(H * (dc[np.newaxis].T / Hs))[:, 0]
            dv[:] = np.asarray(H * (dv[np.newaxis].T / Hs))[:, 0]
        # gradient of obj
        grad[:] = dc
    f = ((Emin + x ** penal * (Emax - Emin)) * ce).sum()
    return f

def Constraint1(x, grad):
    if grad.size > 0:
        p = para()
        nelx = p.nelx
        nely = p.nely
        nelz = p.nelz
        # gradient of con
        dg = list(1 / (0.5 * nelx * nely * nelz + 1e-2) for i in range(nelx * nely * nelz))
        grad[:] = np.array(dg)
    return x.sum()/(0.5 * nelx * nely * nelz + 1e-2)-1

def Constraint2(x, grad):
    if grad.size > 0:
        p = para()
        nelx = p.nelx
        nely = p.nely
        nelz = p.nelz
        # gradient of con
        dg = list(-1 / (0.5 * nelx * nely * nelz - 1e-2) for i in range(nelx * nely * nelz))
        grad[:] = np.array(dg)
    return -x.sum()/(0.5 * nelx * nely * nelz - 1e-2)+1

# MAIN DRIVER
def main():
    # Default input parameters
    p=para()
    nelx = p.nelx
    nely = p.nely
    nelz = p.nelz
    volfrac = p.volfrac
    rmin = p.rmin
    penal = p.penal
    ft = p.ft # ft==0 -> sens, ft==1 -> dens
    # Max and min stiffness
    Emin = 1e-3
    Emax = 1.0
    # Set loop counter and gradient vectors
    loop = 0
    change = 1
    # Allocate design variables (as array), initialize and allocate sens.
    xx = volfrac * np.ones(nely * nelx * nelz, dtype=float)

    # Initialize plot and plot the initial design
    # plot a cross section perpendicular to z direction
    plt.ion()  # Ensure that redrawing is possible
    for i in range(nelz):
        locals()['fig' + str(i+1)], locals()['ax'+str(i+1)] = plt.subplots()
        locals()['im' + str(i+1)] = locals()['ax'+str(i+1)].imshow(-xx[nelx*nely*i:nelx*nely*(i+1)].reshape((nelx, nely)).T, cmap='gray', \
                     interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
        locals()['fig' + str(i + 1)].show()
        plt.show()

    t1 = 0

    while change > 0.01 and loop < 2000:

        t1old=t1
        t1 = time.clock()

        xxold = xx.copy()
        loop = loop + 1

        # use pyOpt to solve the problem
        opt = nlopt.opt(nlopt.LD_MMA, nelx * nely * nelz)
        # minimize the objective function
        opt.set_min_objective(objfunc)
        # bond constraints
        opt.set_lower_bounds(Emin)
        opt.set_upper_bounds(Emax)
        # Nonlinear constraints
        opt.add_inequality_constraint(Constraint1, 1e-4)
        opt.add_inequality_constraint(Constraint2, 1e-4)
        # Stopping criteria
        opt.set_ftol_rel(1e-4)
        # Performing the optimization
        x = opt.optimize(xx)

        t2=time.clock()

        # Filter design variables
        # Finalize assembly and convert to csc format
        HH = FILTERMATRIX(nelx, nely, nelz, rmin)
        H = HH.assembly()
        Hs = H.sum(1)
        if ft == 0:
            xx = x
        elif ft == 1:
            xx = np.asarray(H * x[np.newaxis].T / Hs)[:, 0]
        obj = opt.last_optimum_value()
        vol = xx.sum() / (nelx * nely * nelz)
        #Compute the change by the inf. norm
        change = np.linalg.norm(xx.reshape(nelx * nely * nelz, 1) - xxold.reshape(nelx * nely * nelz, 1), np.inf)

        t3=time.clock()

        # Plot to screen
        for i in range(nelz):
            locals()['im' + str(i + 1)].set_array(-x[nelx*nely*i:nelx*nely*(i+1)].reshape((nelx, nely)).T)
            locals()['fig' + str(i + 1)].canvas.draw()

        t4=time.clock()

        # Write iteration history to screen (req. Python 2.6 or newer)
        print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format( \
            loop, obj, vol, change))
        print("t1-t1old.: {0:.5f} , t2-t1.: {1:.5f} , t3-t2.: {2:.5f} , t3-t2.: {3:.5f}". \
            format(t1-t1old, t2-t1, t3-t2, t4-t3))
    # Make sure the plot stays and that the shell remains
    plt.show()
    raw_input("Press any key...")

#element stiffness matrix
def lk():
    nu = 0.3
    A = np.array([[32,6,-8,6,-6,4,3,-6,-10,3,-3,-3,-4,-8],[-48,0,0,-24,24,0,0,0,12,-12,0,12,12,12]])
    b = np.array([[1],[nu]])
    k = 1/float(144)*np.dot(A.T,b).flatten()

    K1 = np.array([[k[0],k[1],k[1],k[2],k[4],k[4]],
    [k[1],k[0],k[1],k[3],k[5],k[6]],
    [k[1],k[1],k[0],k[3],k[6],k[5]],
    [k[2],k[3],k[3],k[0],k[7],k[7]],
    [k[4],k[5],k[6],k[7],k[0],k[1]],
    [k[4],k[6],k[5],k[7],k[1],k[0]]])

    K2 = np.array([[k[8],k[7],k[11],k[5],k[3],k[6]],
    [k[7],k[8],k[11],k[4],k[2],k[4]],
    [k[9],k[9],k[12],k[6],k[3],k[5]],
    [k[5],k[4],k[10],k[8],k[1],k[9]],
    [k[3],k[2],k[4],k[1],k[8],k[11]],
    [k[10],k[3],k[5],k[11],k[9],k[12]]])

    K3 = np.array([[k[5],k[6],k[3],k[8],k[11],k[7]],
    [k[6],k[5],k[3],k[9],k[12],k[9]],
    [k[4],k[4],k[2],k[7],k[11],k[8]],
    [k[8],k[9],k[1],k[5],k[10],k[4]],
    [k[11],k[12],k[9],k[10],k[5],k[3]],
    [k[1],k[11],k[8],k[3],k[4],k[2]]])

    K4 = np.array([[k[13],k[10],k[10],k[12],k[9],k[9]],
    [k[10],k[13],k[10],k[11],k[8],k[7]],
    [k[10],k[10],k[13],k[11],k[7],k[8]],
    [k[12],k[11],k[11],k[13],k[6],k[6]],
    [k[9],k[8],k[7],k[6],k[13],k[10]],
    [k[9],k[7],k[8],k[6],k[10],k[13]]])

    K5 = np.array([[k[0],k[1],k[7],k[2],k[4],k[3]],
    [k[1],k[0],k[7],k[3],k[5],k[10]],
    [k[7],k[7],k[0],k[4],k[10],k[5]],
    [k[2],k[3],k[4],k[0],k[7],k[1]],
    [k[4],k[5],k[10],k[7],k[0],k[7]],
    [k[3],k[10],k[5],k[1],k[7],k[0]]])

    K6 = np.array([[k[13],k[10],k[6],k[12],k[9],k[11]],
    [k[10],k[13],k[6],k[11],k[8],k[1]],
    [k[6],k[6],k[13],k[9],k[1],k[8]],
    [k[12],k[11],k[9],k[13],k[6],k[10]],
    [k[9],k[8],k[1],k[6],k[13],k[6]],
    [k[11],k[1],k[8],k[10],k[6],k[13]]])

    KE1=np.hstack((K1,K2,K3,K4))
    KE2=np.hstack((K2.T,K5,K6,K3.T))
    KE3=np.hstack((K3.T,K6,K5.T,K2.T))
    KE4=np.hstack((K4,K3,K2,K1.T))
    KE = 1/float(((nu+1)*(1-2*nu)))*np.vstack((KE1,KE2,KE3,KE4))

    return(KE)

# The real main driver
if __name__ == "__main__":
	main()