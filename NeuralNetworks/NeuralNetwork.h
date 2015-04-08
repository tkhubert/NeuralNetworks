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

template<typename ActFunc, typename CtFunc>
class NeuralNetwork
{
public:
    NeuralNetwork() : nbLayers(0) {}
    void addLayer(Layer<ActFunc>& layer);
    
private:
    int    nbLayers;
    
    CtFunc CFunc;
    std::vector<Layer<ActFunc> > layers;
    
};

#endif /* defined(__NeuralNetworks__NeuralNetwork__) */
