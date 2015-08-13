//
//  FCLayer.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 04/07/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef __NeuralNetworks__FCLayer__
#define __NeuralNetworks__FCLayer__

#include "Layer.h"

namespace NN {
    
//
class FCLayer : public Layer
{
public:
    FCLayer(size_t inputSize, size_t outputSize, float dropRate, const ActivationFunc& AFunc)
        : Layer(inputSize, outputSize, dropRate, AFunc) {};
    
    string getName()    const {return "FCLayer";}
    string getDetails() const {return "";}
    
    void fwdProp();
    void bwdProp();
    void calcGrad();
};

}
#endif /* defined(__NeuralNetworks__FCLayer__) */
