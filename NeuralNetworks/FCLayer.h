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
    FCLayer(size_t size, real dropRate, const ActivationFunc& AFunc);
    
    string getName()      const override {return "FCLayer";}
    string getDetails()   const override {return "";}
    LayerClass getClass() const override {return LayerClass::FCLayer;}
    
    void setPrevLayer(Layer* prev) override ;
    void fwdProp()  override;
    void bwdProp()  override;
    void calcGrad() override;
};

}
#endif /* defined(__NeuralNetworks__FCLayer__) */
