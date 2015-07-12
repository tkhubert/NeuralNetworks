//
//  NeuralNetwork.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef __NeuralNetworks__NeuralNetwork__
#define __NeuralNetworks__NeuralNetwork__

#include "includes.h"
#include "CostFunc.h"
#include "Optimizer.h"
#include "Layer.h"
#include "FCLayer.h"
#include "Data.h"

class NeuralNetwork
{
public:
    NeuralNetwork(const CostFunc& _CFunc, const Optimizer& _Optim, std::vector<Layer*>& _layers);
    
    std::string getName()    const;
    std::string getDetails() const;
    
    float getCost()        const {return cost;}
    float getErrRate()     const {return errRate;}
    
    void train(const DataContainer& data);
    void test (const std::vector<LabelData>& lData);
    
    const std::vector<float>& predict(const LabelData& lD);
    
private:
    // members
    size_t inputSize;
    size_t outputSize;
    size_t nbLayers;
    
    const CostFunc&     CFunc;
    const Optimizer&    Optim;
    std::vector<Layer*> layers;
    
    float               cost;
    float               errRate;
    
    std::ofstream       debugFile;
    
    // methods
    void  setInput(const LabelData& lD);
    void  setInput(std::vector<LabelData>::const_iterator dataStart, std::vector<LabelData>::const_iterator dataEnd);
    const std::vector<float>& getOutput() const {return layers[nbLayers-1]->getA();}
    
    size_t isCorrect(std::vector<LabelData>::const_iterator dataStart, std::vector<LabelData>::const_iterator dataEnd) const;
    float  calcCost (std::vector<LabelData>::const_iterator dataStart, std::vector<LabelData>::const_iterator dataEnd) const;
    void   calcDCost(std::vector<LabelData>::const_iterator dataStart, std::vector<LabelData>::const_iterator dataEnd, std::vector<float>& dC);
    void   setDCost (const std::vector<float>& dc) { return layers[nbLayers-1]->setDCost(dc);}
    
    void setNbData(size_t nbData);
    void updateParams();
    void fwdProp(const LabelData& lD);
    void fwdProp(std::vector<LabelData>::const_iterator dataStart, std::vector<LabelData>::const_iterator dataEnd);
    void bwdProp(const std::vector<float>& dC);
    
    void checkGradient();
};

#endif /* defined(__NeuralNetworks__NeuralNetwork__) */
