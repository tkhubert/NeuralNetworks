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
#include "Data.h"

class NeuralNetwork
{
public:
    NeuralNetwork(const CostFunc& _CFunc, const Optimizer& _Optim, std::vector<Layer*>& _layers);
    
    std::string getName()    const;
    std::string getDetails() const;
    
    double getCost()    const {return cost;}
    double getErrRate() const {return errRate;}
    
    void train(const DataContainer& data);
    void test (const std::vector<std::vector<double> >& inputs, const std::vector<int>& labels);
    
    const std::vector<double>& predict(const std::vector<double> & inputs);
    
private:
    // members
    size_t inputSize;
    size_t outputSize;
    size_t nbLayers;
    
    const CostFunc&     CFunc;
    const Optimizer&    Optim;
    std::vector<Layer*> layers;
    
    double               cost;
    double               errRate;
    
    std::ofstream        debugFile;
    
    // methods
    void setInput(const std::vector<double>& input) { layers[0]->setA(input);}
    const std::vector<double>& getOutput() const {return layers[nbLayers-1]->getA();}
    
    bool   isCorrect(int label) const;
    double calcCost (int label) const         { return CFunc.f (getOutput(), label);}
    void calcDCost(int label, std::vector<double>& dc)     { return CFunc.df(getOutput(), label, dc); }
    void   setDCost (const std::vector<double>& dc) { return layers[nbLayers-1]->setDCost(dc);}
    
    void initParams();
    void updateParams();
    void fwdProp(const std::vector<double>& input);
    void bwdProp(const std::vector<double>& dc);
    
    void checkGradient();
};

#endif /* defined(__NeuralNetworks__NeuralNetwork__) */
