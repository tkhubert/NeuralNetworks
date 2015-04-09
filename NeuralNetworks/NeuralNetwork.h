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
#include "Layer.h"


class NeuralNetwork
{
public:
    NeuralNetwork(const CostFunc& _CFunc) : nbLayers(0) , CFunc(_CFunc) {}
    std::string getDetails() const;
    
    void setInput(const std::vector<double>& input) { layers[0].setA(input);}
    void addLayer(Layer& layer);
    
    const std::vector<double>& predict(const std::vector<double> & inputs);
    void train(const std::vector<std::vector<double> >& inputs, const std::vector<double>& labels);
    void test (const std::vector<std::vector<double> >& inputs, const std::vector<double>& labels) const;
    
private:
    // members
    size_t nbLayers;
    
    const CostFunc&    CFunc;
    std::vector<Layer> layers;
    
    // methods
    void fwdProp();
    void bwdProp();
};

#endif /* defined(__NeuralNetworks__NeuralNetwork__) */
