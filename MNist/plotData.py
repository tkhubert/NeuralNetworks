__author__ = 'tkhubert'

import csv
import matplotlib.pyplot as plt

def getFileName(size, CFunc, AFunc, learningRate, lbda, batchSize, nbEpoch):
    filename    = ""
    for s in size:
        filename += str(s) + "_"
    filename   += CFunc+"_"+AFunc+"_"
    filename   += str(learningRate)+"_"+str(lbda)+"_"+str(batchSize)+"_"+str(nbEpoch)
    inputFile   = filename + '.csv'
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
    name = ''
    minErr  = []
    minCost = []
    for lbda in lbdaV:
        mErr  = []
        mCost = []
        for lR in learningRateV:
            filename, inputFile = getFileName(size, CFunc, AFunc, lR, lbda, batchSize, nbEpoch)
            mErr.append(min(getDataW(inputFile, 'test', 'err')))
            mCost.append(min(getDataW(inputFile, 'test', 'cost')))
        minErr.append(mErr)
        minCost.append(mCost)
    
    plot('learningRate', lbdaV, learningRateV, minErr, minCost, name)
    
    print '     ', learningRateV
    for (l,mV) in zip(lbdaV, minErr):
        print 'lambda: ', l, mV
    print
    
def plotGridVslbda(size, CFunc, AFunc, learningRateV, lbdaV, batchSize, nbEpoch):
    name = ''
    minErr  = []
    minCost = []
    for lR in learningRateV:
        mErr  = []
        mCost = []
        for lbda in lbdaV:
            filename, inputFile = getFileName(size, CFunc, AFunc, lR, lbda, batchSize, nbEpoch)
            mErr.append(min(getDataW(inputFile, 'test', 'err')))
            mCost.append(min(getDataW(inputFile, 'test', 'cost')))
        minErr.append(mErr)
        minCost.append(mCost)
    
    plot('lambda', learningRateV, lbdaV, minErr, minCost, name)
    
    print '     ', lbdaV
    for (lR,mV) in zip(learningRateV, minErr):
        print 'lR: ', lR, mV
    print

     
def plotFile(size, CFunc, AFunc, learningRate, lbda, batchSize, nbEpoch):
    filename, inputFile = getFileName(size, CFunc, AFunc, learningRate, lbda, batchSize, nbEpoch)
    
    labels = ['train', 'cross', 'test']
    epoch  = getColumn(inputFile, 0)
    err  = getData(inputFile, 'err')
    cost = getData(inputFile, 'cost')

    plot('epoch', labels, epoch, err, cost, filename)

def plot(xlabel, labels, x, data1, data2, title):
    
    plt.figure(figsize=(18, 9))
    plt.subplot(2,1,1)
    plt.title('Error: '+title)
    for d in data1:
        plt.plot(x, d)
    plt.legend(labels)
    plt.ylabel('Error')
    plt.xlabel(xlabel)
    
    plt.subplot(2,1,2)
    plt.title('Cost: '+title)
    for d in data2:
        plt.plot(x, d)
    plt.legend(labels)
    plt.ylabel('Cost')
    plt.xlabel(xlabel)
    
    plt.savefig(title+'.pdf')
    
    plt.show()
    
def main():
    size = [784, 50, 30, 10]
    CFunc = 'CECFunc'
    AFunc = 'SigAFunc'
    learningRate = 0.2
    lbda         = 3
    batchSize    = 10
    nbEpoch      = 65
    
    lRV = [0.1, 0.2]
    lbdaV = [3, 5]
    plotFile(size, CFunc, AFunc, learningRate, lbda, batchSize, nbEpoch)
    plotVsLR(size, CFunc, AFunc, lRV, lbda, batchSize, nbEpoch)
    plotVsLambda(size, CFunc, AFunc, learningRate, lbdaV, batchSize, nbEpoch)
    plotGridVslR(size, CFunc, AFunc, lRV, lbdaV, batchSize, nbEpoch)
    plotGridVslbda(size, CFunc, AFunc, lRV, lbdaV, batchSize, nbEpoch)
    
if __name__ == '__main__':
    main()