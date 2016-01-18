#!/usr/bin/python

# Python implementation of machine-learned coarse-grained potential(force)
# for PS chains model in melts
# Chan LIU

####################

import os
import sys
import numpy as np
from numpy import linalg as LA
import random
import math
from math import *
import scipy
from scipy import stats
from scipy.spatial.distance import pdist, cdist, squareform
import argparse
import pickle

# default arguments

# Import pickle file or not(True or False)
imprt = False
# Bead type of Training('', 'A' or 'B')
trainType = 'B'
# Maximum size of Coulomb matrix
maxNeighbors = 5
# Fraction of training data
fracTraining = 0.75
# Kernel width
sigma = 10
# Regularization parameter
lambdareg = 0.1

# Number of snapshots
trajNmb = 301
# Number of chains
chnNmb = 9
# number of monomers per chain
monNmb = 10
# number of beads per chain
beadChnNmb = 2 * monNmb
# number of atoms per monomer for PS
atMonNmb = 16
# number of atoms per chain for PS
atChnNmb = monNmb * atMonNmb + 2
# number of all atoms in system
atNmb = chnNmb * atChnNmb
# number of all beads in system
beadNmb = chnNmb * beadChnNmb

# Parse command-line options
parser = argparse.ArgumentParser(description='Machine learning of Coarse-grained force', epilog='Chan LIU (2015)')
parser.add_argument('--imp', dest='imprt', type=bool, default=imprt, help='Whether import ML machine from pickle file')
parser.add_argument('--xyz', dest='xyz', type=str, help='Predict Punch output of XYZ file')
parser.add_argument('--sig', dest='sigma', type=float, default=sigma, help='kernel width')
#parser.add_argument('--zet', dest='zeta', type=float, default=zeta, help='Baseline width')
parser.add_argument('--typ', dest='trainType', type=str, default=trainType, help='Bead type of Training')
parser.add_argument('--lam', dest='lambdareg', type=float, default=lambdareg, help='regularization strength')
parser.add_argument('--frt', dest='fracTraining', type=float, default=fracTraining, help='Fraction of training data')
parser.add_argument('--ngh', dest='maxNeighbors', type=int, default=maxNeighbors, help='Maximum number of neighbors in Coulomb matrix')

args = parser.parse_args()

if args.xyz != None and args.imprt == False:
  print "Error. Can't predict XYZ file without imported ML model."
  exit(1)

def main():

    print "# Read atomistic trajectory file..." 
    sys.stdout.flush()
    TRAJ_FILE = open("traj.gro", 'r')
    CG_FILE = open("cg.gro", 'w')
    newpath = 'cg-traj' 
    if not os.path.exists(newpath): os.mkdir(newpath)
    CMatr_FILE = open("CMatr.txt", 'w')
    PrincAxes_FILE = open("PAxes.txt", 'w')
    trajID = []
    TS_CMatr = []
    TS_PAxes = []
    for t in xrange(trajNmb):
        CGN_FILE = open("cg-traj/cg"+str(t)+".gro", 'w')
        SC_atMass,SC_atNucl,SC_atCor,T_INF,box = parseGRO(TRAJ_FILE)
        CMatr_FILE.write("t = " + str(T_INF) + '\n')
        PrincAxes_FILE.write("t = " + str(T_INF) + '\n')
        trajID.append(T_INF)
        S_beadCor,S_beadMass,S_beadNucl,S_beadRes,S_beadName = mapping(SC_atMass,SC_atNucl,SC_atCor)
        outputCGtraj(CG_FILE,CGN_FILE,T_INF,S_beadRes,S_beadName,S_beadCor,box)
        CGN_FILE.close()
        S_CMatr, S_reorder = buildCoulMatrix(CMatr_FILE,S_beadCor,S_beadNucl,S_beadName,box)
        TS_CMatr.append(S_CMatr)
        S_PAxes = princAxes(PrincAxes_FILE,S_beadCor,S_beadName,S_beadMass,box)
        TS_PAxes.append(S_PAxes)
    TRAJ_FILE.close()
    CG_FILE.close()
    CMatr_FILE.close()
    PrincAxes_FILE.close()
    ######## Get Target forces (from atomistic model) in principal axis systems ########
    TS_Frc = getNBFrc()
    TS_Frot = rotate(TS_PAxes,TS_Frc,trajID,S_beadName)
    ######## Get Baseline forces (from coarse-grained model) in principal axis systems ########

    ######## Compare (output tegether) Target forces and Baseline forces ########

    ######## Segregate data into training and test sets ########
    X_train, y_train, princAxes_train, beadName_train, numBeadTyp_train, \
                          X_test, y_test, princAxes_test, beadName_test = \
                        segregateDataSet(TS_CMatr,TS_Frot,TS_PAxes,S_beadName)   
    # Average distance in training set
    avgDist = 0.0
    avgCnt  = 0
    for i in xrange(len(X_train)):
      for j in xrange(len(X_train[i])):
        avgDist += X_train[i][j]
        avgCnt  += 1
    print "# Average distance in training set: {0:7.4f}".format(avgDist/avgCnt)
    sys.stdout.flush()
    ######## Get alpha matrix ########
    if args.imprt == True:
        print "# Reading kernel-ridge regression coefficients from file..."
        sys.stdout.flush()
        with open("alpha_train.pkl", 'rb') as f:
            X_train, y_train, beadName_train, numBeadTyp_train, alpha_train = pickle.load(f)
        print "numBeadTyp_train = ", numBeadTyp_train
        sys.stdout.flush()
    else:
        norm_train = normalizer(beadName_train, numBeadTyp_train)
        alpha_train = trainMachine(X_train,y_train,norm_train)
        with open("alpha_train.pkl", 'wb') as f:
            pickle.dump([X_train, y_train, beadName_train, numBeadTyp_train, alpha_train], f, protocol=2)
    ######## Predict ########
    print "alpha_train = ", alpha_train
    sys.stdout.flush()
    norm_testtrain = normalizer2(beadName_test,beadName_train, numBeadTyp_train)
    y_test_predict = predictRefinement(X_train, X_test, alpha_train, norm_testtrain)
    compareTestPredict(y_test, y_test_predict, beadName_test)
    ######## Compare (output tegether) Target forces and Predict forces of test sets ########
    Predict_FILE = open("Predict"+args.trainType+"_"+str(args.sigma)+"_"+str(args.lambdareg)+".txt", 'w')
    for i in xrange(len(beadName_test)):
        Predict_FILE.write(beadName_test[i]+'\t'+str(y_test[i][0])+'\t'+str(y_test[i][1])+'\t'+str(y_test[i][2])+'\t'+
                          str(y_test_predict[i][0])+'\t'+str(y_test_predict[i][1])+'\t'+str(y_test_predict[i][2])+'\n')
    Predict_FILE.close()
    return

def parseGRO(TRAJ_FILE):
    '''Read the GRO file to get the masses and coordinates of all atoms'''
    FILE_INF	= TRAJ_FILE.readline().split()
    T_INF	= float(FILE_INF[-1])
    atNmb	= int(TRAJ_FILE.readline())
#    print 'atNmb = ',atNmb
    SC_atMass = []
    SC_atNucl = []
    SC_atCor  = []
    for chn in xrange(chnNmb):
        C_atMass = []
        C_atNucl = []
        C_atCor  = []        
        for i in xrange(atChnNmb):
            TRAJ_INF = TRAJ_FILE.readline().split()
            molTyp = TRAJ_INF[0]
            atTyp = TRAJ_INF[1]
            # add atom mass
            if atTyp[0] == 'C':
                atMass = 12.0110
                atNucl = 6
            elif atTyp[0] == 'H':
                atMass = 1.0080
                atNucl = 1
            else:
                raise TypeError('unknown atom type')
            ########
            atID	= int(TRAJ_INF[2])
            atCor	= (float(TRAJ_INF[3]), float(TRAJ_INF[4]), float(TRAJ_INF[5]))
            atVel	= (float(TRAJ_INF[6]), float(TRAJ_INF[7]), float(TRAJ_INF[8]))
            C_atMass.append(atMass)
            C_atNucl.append(atNucl)
            C_atCor.append(atCor)
        SC_atMass.append(C_atMass)
        SC_atNucl.append(C_atNucl)
        SC_atCor.append(C_atCor)
    trajBOX = TRAJ_FILE.readline().split()
    box	= (float(trajBOX[0]),float(trajBOX[1]),float(trajBOX[2]))
    return SC_atMass,SC_atNucl,SC_atCor,T_INF,box
    
def mapping(SC_atMass,SC_atNucl,SC_atCor):
    ''' Mapping from atomistic model to coarse-grained model
    Note: the center of bead A is the center of mass of the CH2-group 
    and the two CH-groups, that are taken with half of their masses'''

    # Compute the centers of mass of bead A...
    AS_Cor  = np.array([])
    for chn in xrange(chnNmb):
        for i in xrange(monNmb):
            if i == 0:
                A_Cor = np.array(SC_atCor[chn][1])
#                print A_Cor
            else:
                A_Mass, A_Nucl, A_MX, A_MY, A_MZ = 0, 0, 0, 0, 0
                for j in [1, 2, 3]: # CH2-group
                    A_Mass += SC_atMass[chn][16*i+j-1]
                    A_Nucl += SC_atNucl[chn][16*i+j-1]
                    A_MX   += SC_atMass[chn][16*i+j-1] * SC_atCor[chn][16*i+j-1][0]
                    A_MY   += SC_atMass[chn][16*i+j-1] * SC_atCor[chn][16*i+j-1][1]
                    A_MZ   += SC_atMass[chn][16*i+j-1] * SC_atCor[chn][16*i+j-1][2]
                for j in [-12, -11, 4, 5]: # two CH-groups ---- half mass
                    A_Mass += 0.5 * SC_atMass[chn][16*i+j-1]                    
                    A_Nucl += 0.5 * SC_atNucl[chn][16*i+j-1]
                    A_MX   += 0.5 * SC_atMass[chn][16*i+j-1] * SC_atCor[chn][16*i+j-1][0]
                    A_MY   += 0.5 * SC_atMass[chn][16*i+j-1] * SC_atCor[chn][16*i+j-1][1]
                    A_MZ   += 0.5 * SC_atMass[chn][16*i+j-1] * SC_atCor[chn][16*i+j-1][2]
                A_Cor  = np.array([A_MX/A_Mass, A_MY/A_Mass, A_MZ/A_Mass])
            AS_Cor = np.append(AS_Cor, A_Cor)
    AS_Cor = AS_Cor.reshape(chnNmb*monNmb,3)
#    print AS_Cor
#    exit()

    # Compute the centers of mass of bead B...
    BS_Cor  = np.array([])
    for chn in xrange(chnNmb):
        for i in xrange(monNmb):
            B_Mass, B_Nucl, B_MX, B_MY, B_MZ = 0, 0, 0, 0, 0
            for j in xrange(6, 17): # phenyl rings --- atom 6-16
                B_Mass += SC_atMass[chn][16*i+j-1]
                B_Nucl += SC_atNucl[chn][16*i+j-1]
                B_MX   += SC_atMass[chn][16*i+j-1] * SC_atCor[chn][16*i+j-1][0]
                B_MY   += SC_atMass[chn][16*i+j-1] * SC_atCor[chn][16*i+j-1][1]
                B_MZ   += SC_atMass[chn][16*i+j-1] * SC_atCor[chn][16*i+j-1][2]
            B_Cor  = np.array([B_MX/B_Mass, B_MY/B_Mass, B_MZ/B_Mass])
            BS_Cor = np.append(BS_Cor, B_Cor)
    BS_Cor = BS_Cor.reshape(chnNmb*monNmb,3)
#    print BS_Cor
#    exit()

    # combine all beads
    S_beadCor = np.array([])
    S_beadMass = []
    S_beadNucl = []
    S_beadName = []
    S_beadRes = []
    for chn in xrange(chnNmb):    
        for i in xrange(monNmb):
            S_beadCor = np.append(S_beadCor, AS_Cor[monNmb*chn+i])
            S_beadMass.append(A_Mass)
            S_beadNucl.append(A_Nucl)
            S_beadRes.append(chn+1)
            S_beadName.append('A' + str(i+1))
            S_beadCor = np.append(S_beadCor, BS_Cor[monNmb*chn+i])
            S_beadMass.append(B_Mass)
            S_beadNucl.append(B_Nucl)
            S_beadRes.append(chn+1)
            S_beadName.append('B' + str(i+1))
    S_beadCor = S_beadCor.reshape(beadNmb,3)
#    print S_beadCor
#    print S_beadName
#    print S_beadRes
#    exit()
    return S_beadCor,S_beadMass,S_beadNucl,S_beadRes,S_beadName

def outputCGtraj(CG_FILE,CGN_FILE,T_INF,S_beadRes,S_beadName,S_beadCor,box):
    CG_FILE.write('{:s}\n'.format('CG-PS Melt Many Chains t= '+str(T_INF)))
    CGN_FILE.write('{:s}\n'.format('CG-PS Melt Many Chains t= '+str(T_INF)))
    CG_FILE.write('{:5d}\n'.format(beadNmb))
    CGN_FILE.write('{:5d}\n'.format(beadNmb))
    for i in xrange(beadNmb):
        CG_FILE.write('{:5d}{:5s}{:5s}{:5d}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}\n'.format(
                      S_beadRes[i], 'PS1', S_beadName[i], i+1, S_beadCor[i][0],  S_beadCor[i][1], S_beadCor[i][2], 0, 0, 0))
        CGN_FILE.write('{:5d}{:5s}{:5s}{:5d}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}\n'.format(
                      S_beadRes[i], 'PS1', S_beadName[i], i+1, S_beadCor[i][0],  S_beadCor[i][1], S_beadCor[i][2], 0, 0, 0))
    CG_FILE.write('{0:10.5f}{1:10.5f}{2:10.5f}\n'.format(box[0], box[1], box[2]))
    CGN_FILE.write('{0:10.5f}{1:10.5f}{2:10.5f}\n'.format(box[0], box[1], box[2]))
    return

def buildCoulMatrix(CMatr_FILE,S_beadCor,S_beadNucl,S_beadName,box):
    '''Build Coulomb matrix ordered by distance to mainAtomID'''
    if beadNmb-8 < args.maxNeighbors:
        raise ValueError('maxNeighbors should not be larger than bead numbers')
    # periodic boundray condition
    S_CMatr = []
    S_reorder = []
    for mainAtomID in xrange(beadNmb):
        # First compute distance of all atoms to mainAtomID   
        distMain = np.zeros(beadNmb)
        for i in xrange(beadNmb):
            # periodic boundray condition
            diffx = S_beadCor[mainAtomID][0] - S_beadCor[i][0]
            diffy = S_beadCor[mainAtomID][1] - S_beadCor[i][1]
            diffz = S_beadCor[mainAtomID][2] - S_beadCor[i][2]
            diffx = min(diffx**2, (diffx-box[0])**2, (diffx+box[0])**2)
            diffy = min(diffy**2, (diffy-box[1])**2, (diffy+box[1])**2)
            diffz = min(diffz**2, (diffz-box[2])**2, (diffz+box[2])**2) 
            distMain[i] = diffx + diffy + diffz
            # Predict nonbonded force --- The Coulomb Matrix should exclude the 1-5 interaction beads. 
            if i / beadChnNmb == mainAtomID / beadChnNmb:
                if i == mainAtomID - 4 or i == mainAtomID - 3 or i  == mainAtomID - 2 or i == mainAtomID - 1 \
                           or i == mainAtomID + 1  or i == mainAtomID + 2 or i == mainAtomID + 3 or i == mainAtomID + 4:
#                    print mainAtomID,i
                    distMain[i] = box[0]**2 + box[1]**2 + box[2]**2
#        print distMain
#        exit()           
        # Ordered by the distance
        reorder = np.argsort(distMain)
        beadCorOrd = np.zeros((args.maxNeighbors,3))
        for i in xrange(args.maxNeighbors):
            beadCorOrd[i,:] = S_beadCor[reorder[i],:]
        beadNuclOrd = []
        for i in xrange(args.maxNeighbors):
            beadNuclOrd.append(S_beadNucl[reorder[i]]) 
        # compute distance matrix
        d = np.zeros((args.maxNeighbors, args.maxNeighbors))
        for i in xrange(args.maxNeighbors):
            for j in xrange(3):
                diff2 = beadCorOrd[i,j] - beadCorOrd[:,j]
                diff2 **= 2
                d[i,:] += diff2
        np.sqrt(d,d)
        # compute Coulomb matrix
        for i in xrange(args.maxNeighbors):
            for j in xrange(args.maxNeighbors):
                if i != j:
                    d[i,j] = beadNuclOrd[i]*beadNuclOrd[j]/d[i,j]
                else:
                    d[i,i] = 0.5*beadNuclOrd[i]**(2.4)
        O_CMatr = (S_beadName[mainAtomID], d)
        CMatr_FILE.write(str(O_CMatr)+'\n')
        S_CMatr.append(d[np.triu_indices(args.maxNeighbors)])
        S_reorder.append(reorder)
    return S_CMatr, S_reorder

def princAxes(PrincAxes_FILE,S_beadCor,S_beadName,S_beadMass,box):
    ''' Get the principal axis system for mainAtomID'''

    if beadNmb-8 < args.maxNeighbors:
        raise ValueError('maxNeighbors should not be larger than bead numbers')
    # periodic boundray condition
    S_PAxes = []
    S_reorder = []
    for mainAtomID in xrange(beadNmb):
        # First compute distance of all atoms to mainAtomID   
        distMain = np.zeros(beadNmb)
        for i in xrange(beadNmb):
            # periodic boundray condition
            diffx = S_beadCor[mainAtomID][0] - S_beadCor[i][0]
            diffy = S_beadCor[mainAtomID][1] - S_beadCor[i][1]
            diffz = S_beadCor[mainAtomID][2] - S_beadCor[i][2]
            diffx = min(diffx**2, (diffx-box[0])**2, (diffx+box[0])**2)
            diffy = min(diffy**2, (diffy-box[1])**2, (diffy+box[1])**2)
            diffz = min(diffz**2, (diffz-box[2])**2, (diffz+box[2])**2) 
            distMain[i] = diffx + diffy + diffz        
        # Ordered by the distance
        reorder = np.argsort(distMain)
        beadCorOrd = np.zeros((args.maxNeighbors,3))
        for i in xrange(args.maxNeighbors):
            beadCorOrd[i,:] = S_beadCor[reorder[i],:]
        beadMassOrd = []
        for i in xrange(args.maxNeighbors):
            beadMassOrd.append(S_beadMass[reorder[i]])
#        print beadCorOrd,beadMassOrd

        coords = beadCorOrd - S_beadCor[mainAtomID]
        coords2 = sum(np.dot(beadMassOrd,coords*coords)) * np.identity(3)
        inertia = coords2 - np.dot(beadMassOrd * coords.transpose(),coords)
        eigvals,eigvecs = np.linalg.eig(inertia)
        eigorder = np.argsort(-eigvals)
        eigvalsOrd = np.zeros(3)
        eigvecsOrd = np.zeros((3,3))
        eigvalsOrd[:] = eigvals[eigorder[:]]
        eigvecsOrd[:] = eigvecs[eigorder[:]]
#        print S_beadName[mainAtomID],eigvals,eigvecs,eigvalsOrd,eigvecsOrd
        # center of mass of five beads                                                                    
        com = sum([beadMassOrd[j]*beadCorOrd[j] for j in xrange(args.maxNeighbors)]) / sum(beadMassOrd)
        # Orient first two eigenvectors so that com is in positive quadrant.                               
        for v in xrange(3):
            if np.dot(com - S_beadCor[mainAtomID],eigvecsOrd[v]) < 0.:
                eigvecsOrd[v] *= -1.   
        S_PAxes.append(eigvecsOrd)
        O_PAxes = (S_beadName[mainAtomID], eigvecsOrd)
        PrincAxes_FILE.write(str(O_PAxes)+'\n')
#    print S_PAxes
#    exit()
    return S_PAxes

def getNBFrc():
    '''Read the force file to get the nonbonded forces (x, y, z) of all beads '''

    print "# Get the nonbonded forces of beads..." 
    sys.stdout.flush()

    nbF_FILE = open("nbF.txt", 'r')

    TS_Frc = []
    for t in xrange(trajNmb):
        T_INF = nbF_FILE.readline()
        S_beadFrc = []
        for i in xrange(beadNmb):
            nbF_INF = nbF_FILE.readline().split()
            beadTyp = nbF_INF[0]
            beadFrc = np.array([float(nbF_INF[1]), float(nbF_INF[2]), float(nbF_INF[3])])
            S_beadFrc.append(beadFrc)
#        print S_beadFrc
#        exit()
        TS_Frc.append(S_beadFrc)
    nbF_FILE.close()
    return TS_Frc

def rotate(TS_PAxes,TS_Frc,trajID,S_beadName):
    '''Rotate the force into the Principal Axes systems (x', y', z') '''
    print "# Rotate the force vector of beads..."
    sys.stdout.flush()
    Frot_FILE = open("Frot.txt", 'w')
    TS_Frot = []
    for t in xrange(trajNmb):
        Frot_FILE.write('t = '+str(trajID[t])+'\n')
        S_Frot  = []
        N_Frot  = []
        for i in xrange(beadNmb):
            Frot = np.dot(TS_PAxes[t][i], TS_Frc[t][i])
            Frot_FILE.write(S_beadName[i]+'\t'+str(Frot[0])+'\t'+str(Frot[1])+'\t'+str(Frot[2])+'\n')
            S_Frot.append(Frot)
            N_Frot.append(LA.norm(Frot))
        TS_Frot.append(S_Frot)
    Frot_FILE.close()
    return TS_Frot

def segregateDataSet(TS_CMatr,TS_Frot,TS_PAxes,S_beadName):
    '''Segregate data into training and test sets.'''

    print "# Build flatten lists..."
    sys.stdout.flush()
    beadNmb = len(S_beadName)
    beadNameFlat = [S_beadName[monNmb*2*chn+i] for t in xrange(trajNmb) for chn in xrange(chnNmb) for i in xrange(1,monNmb*2)]
    CMatrFlat = [TS_CMatr[t][monNmb*2*chn+i] for t in xrange(trajNmb) for chn in xrange(chnNmb) for i in xrange(1,monNmb*2)]
    FrotFlat = [TS_Frot[t][monNmb*2*chn+i] for t in xrange(trajNmb) for chn in xrange(chnNmb) for i in xrange(1,monNmb*2)]
    princAxesFlat = [TS_PAxes[t][monNmb*2*chn+i] for t in xrange(trajNmb) for chn in xrange(chnNmb) for i in xrange(1,monNmb*2)]
    dataSize = len(beadNameFlat)
    numBeadTyp = {'A':0, 'B':0,}
    numBeadTyp['A'] = sum(1 for name in beadNameFlat if name[0] == 'A')
    numBeadTyp['B'] = sum(1 for name in beadNameFlat if name[0] == 'B')
    print "numBeadTyp = ", numBeadTyp
    sys.stdout.flush()

    print "# Segregate data into training and test sets..."
    sys.stdout.flush()
    # Training set
    eleChosen = []                                                                               
    X_train = []
    y_train = []
    beadName_train = []
    princAxes_train = []
    filterFracTraining = args.fracTraining
    if args.trainType != '':
        filterFracTraining = args.fracTraining * numBeadTyp[args.trainType]/dataSize
    while len(eleChosen)*1./dataSize < filterFracTraining:
        rnd = random.randint(0,dataSize-1)
        if rnd not in eleChosen:
            chooseData = True
            if args.trainType != '':
                chooseData = False
                if beadNameFlat[rnd][0] == args.trainType:
                    chooseData = True
            if chooseData:
                eleChosen.append(rnd)
                X_train.append(CMatrFlat[rnd])
                y_train.append(FrotFlat[rnd])
                princAxes_train.append(princAxesFlat[rnd])
                beadName_train.append(beadNameFlat[rnd])
    numBeadTyp_train = {'A':0, 'B':0,}
    numBeadTyp_train['A'] = sum(1 for name in beadName_train if name[0] == 'A')
    numBeadTyp_train['B'] = sum(1 for name in beadName_train if name[0] == 'B')
    print "numBeadTyp_train = ", numBeadTyp_train
    sys.stdout.flush()

    # The rest is the test set                                                      
    X_test = []
    y_test = []
    beadName_test = []
    princAxes_test = []                                                                 
    for i in xrange(dataSize):
        if i not in eleChosen:
            chooseData = True
            if args.trainType != '':
                chooseData = False
                if beadNameFlat[i][0] == args.trainType:
                    chooseData = True
            if chooseData:
                X_test.append(CMatrFlat[i])
                y_test.append(FrotFlat[i])
                princAxes_test.append(princAxesFlat[i])
                beadName_test.append(beadNameFlat[i])
    return X_train, y_train, princAxes_train, beadName_train, numBeadTyp_train, \
                                 X_test, y_test, princAxes_test, beadName_test

def normalizer(beadName1, numBeadTyp):
    norm = np.zeros((len(beadName1),len(beadName1)))
    for i in xrange(len(beadName1)):
        for j in xrange(len(beadName1)):
            norm[i,j] = sqrt(numBeadTyp[beadName1[i][0]]*numBeadTyp[beadName1[j][0]])
    return norm

def normalizer2(beadName1,beadName2, numBeadTyp):
    norm = np.zeros((len(beadName1),len(beadName2)))
    for i in xrange(len(beadName1)):
        for j in xrange(len(beadName2)):
            norm[i,j] = sqrt(numBeadTyp[beadName1[i][0]]*numBeadTyp[beadName2[j][0]])
    return norm

def trainMachine(X,y,norm_train):
    '''Train kernel ridge regression model --- Laplacian kernels.
    Return alpha matrix of coefficients.'''
    print "# Training algorithm..."
    print "#   building kernel matrix of size (" + str(len(X)) + "," + str(len(X)) + ")"
    sys.stdout.flush()
    pairwise_dists = squareform(pdist(X, 'cityblock'))/norm_train
    kmatr = scipy.exp(- pairwise_dists / args.sigma)
    kmatr += args.lambdareg * np.identity(len(y))
    print "#   solving for alpha coefficients"
    sys.stdout.flush()
    alpha_train = LA.solve(kmatr,y)
    print "#   training finished."
    sys.stdout.flush()
    return alpha_train

def predictRefinement(X_train, X_test, alpha_train, norm_testtrain):
    '''Use kernel ridge regression model to predict --- Laplacian kernels.'''
    pairwise_dists = cdist(X_test, X_train, 'cityblock')/norm_testtrain
    kmatr = scipy.exp(- pairwise_dists / args.sigma)
    y_test_predict = np.dot(kmatr, alpha_train)
    return y_test_predict

def compareTestPredict(y_test, y_predict, beadName_test):
    if len(y_predict) != len(y_test):
        print "Error: prediction and test sets of different lengths."
        sys.stdout.flush()
    pred = np.array(y_predict)
    test = np.array(y_test)
    # Compute R^2 and MAE
    r2 = np.zeros(3)
    mae = np.zeros(3)
    for i in xrange(3):
        r2[i] = stats.pearsonr(pred[:,i],test[:,i])[0]
        # MAE                                                                                              
        mae_i = 0.0
        for j in xrange(len(pred)):
            mae_i += abs(pred[j][i]-test[j][i])                                                       
        mae_i /= len(pred)
        mae[i] = mae_i
    r2Comb  = sum(r2[:])/3.
    maeComb = sum(mae[:])/3.
    print "R^2Comb = ", r2Comb
    print "MAEComb = ", maeComb
    print "R^2 = ", r2
    print "MAE = ", mae
    sys.stdout.flush()
    return

if __name__== "__main__":
    main()
