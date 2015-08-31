//
//  ConvLayer.cpp
//  NeuralNetworks
//
//  Created by Thomas Hubert on 15/08/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#include "ConvLayer.h"

namespace NN
{
    
ConvLayer::ConvLayer(size_t width, size_t height, size_t depth, size_t mapSize, size_t stride, const ActivationFunc& AFunc) :
    width (width),
    height(height),
    depth (depth),
    mapSize(mapSize),
    stride(stride),
    Layer(width*height*depth, 0, AFunc)
{
    auto biasSize = depth;
    bias.resize(biasSize);
    dbias.resize(biasSize);
    vbias.resize(biasSize);
};
//
void ConvLayer::setPrevLayer(Layer* prev)
{
    prevLayer = prev;
    inputSize = prevLayer->getOutputSize();
    
    assert(prevLayer->getClass() == LayerClass::ConvLayer);
    auto prevDepth  = static_cast<ConvLayer*>(prevLayer)->getDepth();
    auto weightSize = mapSize*mapSize*prevDepth;
    
    weight.resize (weightSize);
    dweight.resize(weightSize);
    vweight.resize(weightSize);
    
    initParams();
}
//
void ConvLayer::fwdProp()
{
    ConvLayer* prevConvLayer = static_cast<ConvLayer*>(prevLayer);
    const auto& prevA = prevConvLayer->getA();
    auto prevHeight   = prevConvLayer->getHeight();
    auto prevWidth    = prevConvLayer->getWidth();
    auto prevDepth    = prevConvLayer->getDepth();
    
    for (size_t ode=0; ode<depth; ++ode)
    {
        for (size_t oh=0; oh<height; ++oh)
        {
            auto ihMin = oh-(mapSize-1)/2;
            
            for (size_t ow=0; ow<width; ++ow)
            {
                auto iwMin = ow-(mapSize-1)/2;
        
                auto val = bias[ode];
                
                for (size_t ide=0; ide<prevDepth; ++ide)
                {
                    for (size_t ih=0; ih<mapSize; ++ih)
                    {
                        for (size_t iw=0; iw<=mapSize; ++iw)
                        {
                            auto weightIdx = ide*mapSize*mapSize+ih*mapSize+iw;
                            auto prevAIdx  = ide*prevWidth*prevHeight+(ihMin+ih)*prevHeight+(iwMin+iw);
                            val += weight[weightIdx]*prevA[prevAIdx];
                        }
                    }
                }
                
                auto aIdx = ode*width*height+oh*width+ow;
                a[aIdx] = AFunc.f(val);
            }
        }
    }
}
//
    void ConvLayer::bwdProp()
    {
        
    }
    //
    void ConvLayer::calcGrad()
    {
        
    }
    
        
    
}
