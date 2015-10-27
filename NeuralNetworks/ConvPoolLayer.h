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
    
    string getName()      const override {return "ConvPoolLayer";}
    string getDetails()   const override {return "";}
    LayerClass getClass() const override {return LayerClass::ConvPoolLayer;}
    
    void setPrevLayer(Layer* layer) override;
    void fwdProp()  override;
    void bwdProp()  override;
    void calcGrad() override {};
    
private:
    vec_r maxIdx;
    
    void resize(size_t nbData) override;
    void validate() const;
};
    
}

#endif /* defined(__NeuralNetworks__ConvPoolLayer__) */
