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

class NeuralNetwork
{
public:
    NeuralNetwork(const CostFunc& _CFunc, const Optimizer& _Optim) : nbLayers(0) , CFunc(_CFunc), Optim(_Optim) {}
    std::string getDetails() const;
    
    void setInput(const std::vector<double>& input) { layers[0]->setA(input);}
    void addLayer(Layer& layer);
    
    const std::vector<double>& getOuptut() const {return layers[nbLayers-1]->getA();}
    
    double getCost() const {return c;}
    
    const std::vector<double>& predict(const std::vector<double> & inputs);
    void train(const std::vector<std::vector<double> >& inputs, const std::vector<std::vector<double> >& labels);
    void test (const std::vector<std::vector<double> >& inputs, const std::vector<std::vector<double> >& labels);
    
private:
    // members
    size_t nbLayers;
    
    const CostFunc&     CFunc;
    const Optimizer&    Optim;
    std::vector<Layer*> layers;
    
    double               c;
    
    // methods
    double calcCost (const std::vector<double>& label) const { return CFunc.f (getOuptut(), label);}
    double calcDCost(size_t i, const std::vector<double>& label)
    {
        return CFunc.df(i, getOuptut(), label);
    }
    void   setDCost (std::vector<double>& dc)                                       { return layers[nbLayers-1]->setDCost(dc);}
    
    void fwdProp();
    void bwdProp();
};

#endif /* defined(__NeuralNetworks__NeuralNetwork__) */
