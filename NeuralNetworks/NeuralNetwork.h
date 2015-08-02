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
#include "Data.h"

namespace NN {
    
class NeuralNetwork
{
public:
    NeuralNetwork(const CostFunc& CFunc, const Optimizer& Optim, vector<unique_ptr<Layer>>&& layers);
    
    string getName()    const;
    string getDetails() const;
    
    float getCost()    const {return cost;}
    float getErrRate() const {return errRate;}
    
    void train(const DataContainer& data);
    void test (const vector<LabelData>& lData);
    
    const auto& predict(const LabelData& lD);
    
private:
    // members
    size_t inputSize;
    size_t outputSize;
    size_t nbLayers;
    
    const CostFunc&     CFunc;
    const Optimizer&    Optim;
    vector<unique_ptr<Layer>> layers;
    
    float               cost;
    float               errRate;
    
    ofstream       debugFile;
    
    // methods
    void  setInput(const LabelData& lD);
    void  setInput(LabelDataCItr dataStart, LabelDataCItr dataEnd);
    const auto& getOutput() const {return layers[nbLayers-1]->getA();}
    
    size_t isCorrect(LabelDataCItr dataStart, LabelDataCItr dataEnd) const;
    float  calcCost (LabelDataCItr dataStart, LabelDataCItr dataEnd) const;
    void   calcDCost(LabelDataCItr dataStart, LabelDataCItr dataEnd, vector<float>& dC);
    void   setDCost (const vector<float>& dc) { return layers[nbLayers-1]->setDCost(dc);}
    
    void setNbData(size_t nbData);
    void updateParams();
    void fwdProp(const LabelData& lD);
    void fwdProp(LabelDataCItr dataStart, LabelDataCItr dataEnd);
    void bwdProp(const vector<float>& dC);
    
    void checkGradient();
};
    
}

#endif /* defined(__NeuralNetworks__NeuralNetwork__) */
