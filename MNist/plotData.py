__author__ = 'tkhubert'

import numpy
import csv
import matplotlib.pyplot as plt

def getFileName(size, CFunc, AFunc, learningRate, lbda, batchSize, nbEpoch):
    filename    = ""
    for s in size:
        filename += str(s) + "_"
    filename   += CFunc+"_"+AFunc+"_"
    filename   += str(learningRate)+"_"+str(lbda)+"_"+str(batchSize)+"_"+str(nbEpoch)
    inputFile = filename+".csv"
    return [filename, inputFile]

def getColumn(filename, column):
    results = csv.reader(open(filename), delimiter=",")
    return [result[column] for result in results]
     
def getDataW(inputFile, what, type):    
    if (what=='train'):
        idx = 2
    elif (what=='cross'):
    	idx = 3
    elif (what=='test'):
        idx = 4
    else:
	    idx = -10
    if (type=='cost'):
	    idx += 3
    return getColumn(inputFile, idx)

def getData(inputFile, type):
    return [getDataW(inputFile, what, type) for what in ['train', 'cross', 'test']]

def plotVsLambda(size, CFunc, AFunc, learningRate, lbda, batchSize, nbEpoch):
    name = 'vsLambda'
    dataErr = []
    dataCost = []
    for l in lbda:
        filename, inputFile = getFileName(size, CFunc, AFunc, learningRate, l, batchSize, nbEpoch)
        dataErr.append( getDataW(inputFile, 'cross', 'err'))
        dataCost.append(getDataW(inputFile, 'cross', 'cost'))
    
    epoch  = getColumn(inputFile, 0)

    plot('epoch', lbda, epoch, dataErr, dataCost, name)

def plotVsLR(size, CFunc, AFunc, learningRate, lbda, batchSize, nbEpoch):
    name = 'vsLearningRate'
    dataErr = []
    dataCost = []
    for lR in learningRate:
        filename, inputFile = getFileName(size, CFunc, AFunc, lR, lbda, batchSize, nbEpoch)
        dataErr.append( getDataW(inputFile, 'cross', 'err'))
        dataCost.append(getDataW(inputFile, 'cross', 'cost'))
    
    epoch  = getColumn(inputFile, 0)
        
    plot('epoch', learningRate, epoch, dataErr, dataCost, name)

def plotGridVslR(size, CFunc, AFunc, learningRateV, lbdaV, batchSize, nbEpoch):
    name = 'minVsLR'
    minErr  = []
    minCost = []
    for lbda in lbdaV:
        mErr  = []
        mCost = []
        for lR in learningRateV:
            filename, inputFile = getFileName(size, CFunc, AFunc, lR, lbda, batchSize, nbEpoch)
            mErr.append(min(getDataW(inputFile, 'cross', 'err')))
            mCost.append(min(getDataW(inputFile, 'cross', 'cost')))
        minErr.append(mErr)
        minCost.append(mCost)
    
    plot('learningRate', lbdaV, learningRateV, minErr, minCost, name)
    
def plotGridVslbda(size, CFunc, AFunc, learningRateV, lbdaV, batchSize, nbEpoch):
    name = 'minVsLbda'
    minErrCV  = []
    minCostCV = []
    minErrT  = []
    minCostT = []
    for lR in learningRateV:
        mErrCV  = []
        mCostCV = []
        mErrT   = []
        mCostT  = []
        for lbda in lbdaV:
            filename, inputFile = getFileName(size, CFunc, AFunc, lR, lbda, batchSize, nbEpoch)
            crossErr  = getDataW(inputFile, 'cross', 'err')
            crossCost = getDataW(inputFile, 'cross', 'cost')
            testErr  = getDataW(inputFile, 'test', 'err')
            testCost = getDataW(inputFile, 'test', 'cost')
            index = crossCost.index(min(crossCost))
            mErrCV.append(crossErr[index])
            mCostCV.append(crossCost[index])
            mErrT.append(testErr[index])
            mCostT.append(testCost[index])
        minErrCV.append(mErrCV)
        minCostCV.append(mCostCV)
        minErrT.append(mErrT)
        minCostT.append(mCostT)   
         
    plot('lambda', learningRateV, lbdaV, minErrCV, minCostCV, name)
    
    lbdacost = [min(c) for c in minCostCV]
    lbdaIdx  = [c.index(min(c)) for c in minCostCV]
    lRIdx    = lbdacost.index(min(lbdacost))
    filename, inputFile = getFileName(size, CFunc, AFunc, learningRateV[lRIdx], lbdaV[lbdaIdx[lRIdx]], batchSize, nbEpoch)
    crossCost  = getDataW(inputFile, 'cross', 'err')
    index = crossCost.index(min(crossCost))
    
    print 'Best Cost at', learningRateV[lRIdx], lbdaV[lbdaIdx[lRIdx]], str(index)+'/'+str(nbEpoch), ':',
    print minCostCV[lRIdx][lbdaIdx[lRIdx]], minErrCV[lRIdx][lbdaIdx[lRIdx]], "," ,
    print minCostT[lRIdx][lbdaIdx[lRIdx]] , minErrT[lRIdx][lbdaIdx[lRIdx]]

    name = 'minVsLbda'
    minErrCV  = []
    minCostCV = []
    minErrT  = []
    minCostT = []
    x = [[] for i in xrange(len(lbdaV))]
    y = [[] for i in xrange(len(lbdaV))]
    for lR in learningRateV:
        mErrCV  = []
        mCostCV = []
        mErrT   = []
        mCostT  = []
        
        idxLbda = 0
        for lbda in lbdaV:
            filename, inputFile = getFileName(size, CFunc, AFunc, lR, lbda, batchSize, nbEpoch)
            crossErr  = getDataW(inputFile, 'cross', 'err')
            crossCost = getDataW(inputFile, 'cross', 'cost')
            testErr   = getDataW(inputFile, 'test', 'err')
            testCost  = getDataW(inputFile, 'test', 'cost')
            
            for cC, tC, cE, tE in zip(crossCost, testCost, crossErr, testErr):
            	x[idxLbda].append(cC)
            	x[idxLbda].append(tC)
            	y[idxLbda].append(cE)
            	y[idxLbda].append(tE)
            index = crossErr.index(min(crossErr))
            mErrCV.append(crossErr[index])
            mCostCV.append(crossCost[index])
            mErrT.append(testErr[index])
            mCostT.append(testCost[index])
            idxLbda+=1
            
        minErrCV.append(mErrCV)
        minCostCV.append(mCostCV)
        minErrT.append(mErrT)
        minCostT.append(mCostT)   
         
    plot('lambda', learningRateV, lbdaV, minErrCV, minCostCV, name)
    
    lbdacost = [min(c) for c in minErrCV]
    lbdaIdx  = [c.index(min(c)) for c in minErrCV]
    lRIdx    = lbdacost.index(min(lbdacost))
    filename, inputFile = getFileName(size, CFunc, AFunc, learningRateV[lRIdx], lbdaV[lbdaIdx[lRIdx]], batchSize, nbEpoch)
    crossErr  = getDataW(inputFile, 'cross', 'err')
    index = crossErr.index(min(crossErr))
    
    print 'Best Err at', learningRateV[lRIdx], lbdaV[lbdaIdx[lRIdx]], str(index)+'/'+str(nbEpoch), ':',
    print minCostCV[lRIdx][lbdaIdx[lRIdx]], minErrCV[lRIdx][lbdaIdx[lRIdx]], "," ,
    print minCostT[lRIdx][lbdaIdx[lRIdx]] , minErrT[lRIdx][lbdaIdx[lRIdx]]
    print 
    
    for i in range(len(lbdaV)):
    	plt.plot(x[i],y[i], linestyle='None', marker='o')
    plt.legend(lbdaV)
    plt.ylabel('Error')
    plt.xlabel('Cost')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
    return learningRateV[lRIdx], lbdaV[lbdaIdx[lRIdx]]
    
def plotFilename(filename):
    inputFile = filename+'.csv'
    labels = ['train', 'cross', 'test']
    epoch  = getColumn(inputFile, 0)
    err  = getData(inputFile, 'err')
    cost = getData(inputFile, 'cost')

    plot('epoch', labels, epoch, err, cost, filename)
    
def plotFile(size, CFunc, AFunc, learningRate, lbda, batchSize, nbEpoch):
    filename, dummy = getFileName(size, CFunc, AFunc, learningRate, lbda, batchSize, nbEpoch)
    
    plotFilename(filename)

def plot(xlabel, labels, x, data1, data2, title):
    
    plt.figure(figsize=(18, 9))
    plt.subplot(2,1,1)
    plt.title('Error: '+title)
    for d in data1:
    	d1 = [min(0.1,max(0.00001,float(err))) for err in d]
        plt.plot(x, d1)
    plt.legend(labels)
    plt.ylabel('Error')
    plt.xlabel(xlabel)
    plt.yscale('log')
    
    plt.subplot(2,1,2)
    plt.title('Cost: '+title)
    for d in data2:
        d2 = [max(0.00001,float(err)) for err in d]
        plt.plot(x, d2)
    plt.legend(labels)
    plt.ylabel('Cost')
    plt.xlabel(xlabel)
    plt.yscale('log')
    
    plt.savefig(title+'.pdf')
    
    plt.show()

def compareFiles(files):
    epoch  = getColumn(files[0], 0)
    
    err  = []
    cost = []
    for file in files:
        err.append(getData(file, 'err'))
        cost.append(getData(file, 'cost'))
    
    newErr = [[] for x in range(3)]
    newCost = [ [] for x in range(3)]
    for i in range(len(files)):
        for j in range(3):
            newErr[j].append(err[i][j])
            newCost[j].append(cost[i][j])

    titles = ['train', 'cross', 'test']
    labels = [x for x in range(len(files))]
    for j in range(3):
        plot('epoch', labels, epoch, newErr[j], newCost[j], titles[j]) 
    
def main():
    files = []
    ##files.append('784_11520_2880_100_10_SMCFunc_RLAFunc_0.005_0_0_20_100.csv')
    #files.append('784_11520_2880_100_10_SMCFunc_RLAFunc_0.01_0_0_20_100.csv')
    ##files.append('784_11520_2880_100_10_SMCFunc_RLAFunc_0.02_0_0_20_100.csv')
    ##files.append('784_11520_2880_100_10_SMCFunc_RLAFunc_0.001_0.9_0_20_100.csv')
    ##files.append('784_11520_2880_100_10_SMCFunc_RLAFunc_0.002_0.9_0_20_100.csv')
    ##files.append('784_11520_2880_100_10_SMCFunc_RLAFunc_0.005_0.9_0_20_100.csv')
    ##files.append('784_11520_2880_100_10_SMCFunc_RLAFunc_0.0075_0.9_0_20_100.csv')
    #files.append('784_11520_2880_100_10_SMCFunc_RLAFunc_0.01_0.9_0_20_100.csv')
    ##files.append('784_11520_2880_2560_640_100_10_SMCFunc_RLAFunc_0.01_0_0_20_100.csv')
    ##files.append('784_11520_2880_2560_640_100_10_SMCFunc_RLAFunc_0.02_0_0_20_100.csv')
    #files.append('784_11520_2880_2560_640_100_10_SMCFunc_RLAFunc_0.03_0_0_20_100.csv')
    ##files.append('784_11520_2880_2560_640_100_10_SMCFunc_RLAFunc_0.1_0_0_20_100.csv')
    ##files.append('784_11520_2880_2560_640_100_10_SMCFunc_RLAFunc_0.001_0.9_0_20_100.csv')
    files.append('784_11520_2880_2560_640_100_10_SMCFunc_RLAFunc_0.002_0.9_0_20_100.csv')
    files.append('784_11520_2880_2560_640_100_10_SMCFunc_RLAFunc_0.003_0.9_0_20_100.csv')
    ##files.append('784_11520_2880_2560_640_100_10_SMCFunc_RLAFunc_0.005_0.9_0_20_100.csv')
    ##files.append('784_11520_2880_2560_640_100_10_SMCFunc_RLAFunc_0.01_0.9_0_20_100.csv')
    files.append('784_11520_2880_2560_100_10_SMCFunc_RLAFunc_NMOptim_0.001_0.9_0_20_100.csv')
    files.append('784_11520_2880_2560_100_10_SMCFunc_RLAFunc_NMOptim_0.002_0.9_0_20_100.csv')
    files.append('784_11520_2880_2560_100_10_SMCFunc_RLAFunc_NMOptim_0.001_0.9_2_20_100.csv')
    files.append('784_11520_2880_2560_100_10_SMCFunc_RLAFunc_NMOptim_0.002_0.9_2_20_100.csv')
    files.append('784_11520_2880_2560_100_10_SMCFunc_RLAFunc_NMOptim_0.005_0.9_2_20_100.csv')
    compareFiles(files)
	
    size = [784, 100, 100, 10]
    CFunc = 'SMCFunc'
    AFunc = 'RLAFunc'
    batchSize    = 10
    nbEpoch      = 40
    
    lRV   = [0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15]
    lbdaV = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10]

    learningRate, lbda = plotGridVslbda(size, CFunc, AFunc, lRV, lbdaV, batchSize, nbEpoch)
    plotFile      (size, CFunc, AFunc, learningRate, lbda, batchSize, nbEpoch)
    plotVsLR      (size, CFunc, AFunc, lRV         , lbda, batchSize, nbEpoch)
    plotVsLambda  (size, CFunc, AFunc, learningRate, lbdaV, batchSize, nbEpoch)
    
if __name__ == '__main__':
    main()