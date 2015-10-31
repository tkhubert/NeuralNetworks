//
//  NeuralNetwork.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef __NeuralNetworks__NeuralNetwork__
#define __NeuralNetworks__NeuralNetwork__

#include "NN.h"
#include "CostFunc.h"
#include "Optimizer.h"
#include "Layer.h"
#include "Data.h"

namespace NN {
    
class NeuralNetwork
{
public:
    NeuralNetwork(const CostFunc& CFunc, vector<unique_ptr<Layer>>&& layers);
    
    string getName()    const;
    string getDetails() const;
    
    real getCost()    const {return cost;}
    real getErrRate() const {return errRate;}
    
    void train(const DataContainer& data, const Optimizer& Optim);
    void test (const vector<LabelData>& lData, size_t batchSize=20);
    void checkGradient(LabelDataCItr lDStart, LabelDataCItr lDEnd);
    
private:
    // members
    size_t inputSize;
    size_t outputSize;
    size_t nbLayers;
    
    const CostFunc& CFunc;
    vector<unique_ptr<Layer>> layers;
    
    real cost;
    real errRate;
    
    // methods
    void  setInput(LabelDataCItr dataStart, LabelDataCItr dataEnd);
    const auto& getOutput() const {return layers.back()->getA();}
    
    size_t isCorrect(LabelDataCItr dataStart, LabelDataCItr dataEnd) const;
    real   calcCost (LabelDataCItr dataStart, LabelDataCItr dataEnd) const;
    void   setDCost (LabelDataCItr dataStart, LabelDataCItr dataEnd);
    
    void setNbData(size_t nbData);
    void setPhase (Phase phase);
    void setDrop  ();
    
    void fwdProp(LabelDataCItr dataStart, LabelDataCItr dataEnd);
    void bwdProp(LabelDataCItr dataStart, LabelDataCItr dataEnd);
    void calcGrad();
    void regularize(real lambda);
    void updateParams(vector<unique_ptr<Optimizer>>& optim);
    
};
    
}

#endif /* defined(__NeuralNetworks__NeuralNetwork__) */
