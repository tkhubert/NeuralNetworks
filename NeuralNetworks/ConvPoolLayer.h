//
//  ConvPoolLayer.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 05/09/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef __NeuralNetworks__ConvPoolLayer__
#define __NeuralNetworks__ConvPoolLayer__

#include "ConvLayer.h"

namespace NN
{
    
class ConvPoolLayer : public ConvLayer
{
public:
    ConvPoolLayer(size_t width, size_t height, size_t depth, size_t mapSize, size_t stride, const ActivationFunc& AFunc);
    
    string getName()      const {return "ConvPoolLayer";}
    string getDetails()   const {return "";}
    LayerClass getClass() const {return LayerClass::ConvPoolLayer;}
    
    void setPrevLayer(Layer* layer);
    void fwdProp();
    void bwdProp();
    void calcGrad() {};
    
private:
    vec_r maxIdx;
    
    void resize(size_t nbData);
};
    
}

#endif /* defined(__NeuralNetworks__ConvPoolLayer__) */
