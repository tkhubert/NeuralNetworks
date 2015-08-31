//
//  ConvLayer.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 15/08/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef __NeuralNetworks__ConvLayer__
#define __NeuralNetworks__ConvLayer__

#include "Layer.h"

namespace NN
{
    
class ConvLayer : public Layer
{
public:
    ConvLayer(size_t width, size_t height, size_t depth, size_t mapSize, size_t stride, const ActivationFunc& AFunc);
    
    string getName()      const {return "ConvLayer";}
    string getDetails()   const {return "";}
    LayerClass getClass() const {return LayerClass::ConvLayer;}
    
    auto getWidth()   const {return width;}
    auto getHeight()  const {return height;}
    auto getDepth()   const {return depth;}
    auto getMapSize() const {return mapSize;}
    auto getStride()  const {return stride;}
    
    void setPrevLayer(Layer* layer);
    void fwdProp();
    void bwdProp();
    void calcGrad();
    
private:
    size_t width;
    size_t height;
    size_t depth;
    size_t mapSize;
    size_t stride;

};

}

#endif /* defined(__NeuralNetworks__ConvLayer__) */
