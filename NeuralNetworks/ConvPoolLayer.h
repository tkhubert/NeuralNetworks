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
    
    // getters
    string getName()      const override {return "ConvPoolLayer";}
    string getDetails()   const override {return "";}
    LayerClass getClass() const override {return LayerClass::ConvPoolLayer;}
    
    // main methods
    void setFromPrev(const Layer* prevLayer) override;
    void fwdProp    (const Layer* prevLayer) override;
    void bwdProp    (      Layer* prevLayer) override;
    void calcGrad   (const Layer* prevLayer) override {};
    
private:
    vec_r maxIdx;
    
    void resize(size_t nbData) override;
    void validate(const Layer* prevLayer) const override;
};
    
}

#endif /* defined(__NeuralNetworks__ConvPoolLayer__) */
