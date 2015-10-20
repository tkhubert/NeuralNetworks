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
#include "FCLayer.h"
#include "ConvLayer.h"
#include "ConvPoolLayer.h"
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
    
    void train(const DataContainer& data, Optimizer& Optim);
    void test (const vector<LabelData>& lData, size_t batchSize=20);
    void checkGradient(const LabelData& lD);
    
    const auto& predict(const LabelData& lD);
    
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
    void  setInput(const LabelData& lD);
    void  setInput(LabelDataCItr dataStart, LabelDataCItr dataEnd);
    const auto& getOutput() const {return layers[nbLayers-1]->getA();}
    
    size_t isCorrect(LabelDataCItr dataStart, LabelDataCItr dataEnd) const;
    real   calcCost (LabelDataCItr dataStart, LabelDataCItr dataEnd) const;
    void   calcDCost(LabelDataCItr dataStart, LabelDataCItr dataEnd, vec_r& dC);
    void   setDCost (const vec_r& dc) { return layers[nbLayers-1]->setDCost(dc);}
    
    void setNbData(size_t nbData);
    void setPhase (Phase phase);
    void setDrop  ();
    
    void updateParams(Optimizer& optim);
    void fwdProp(const LabelData& lD);
    void fwdProp(LabelDataCItr dataStart, LabelDataCItr dataEnd);
    void bwdProp(const vec_r& dC);
    void calcGrad();
};
    
}

#endif /* defined(__NeuralNetworks__NeuralNetwork__) */
