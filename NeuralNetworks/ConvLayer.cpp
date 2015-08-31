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
    auto weightSize = mapSize*mapSize*depth*prevDepth;
    
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
            auto ihStart = oh-(mapSize-1)/2;
            
            for (size_t ow=0; ow<width; ++ow)
            {
                auto iwStart = ow-(mapSize-1)/2;
        
                auto val = bias[ode];
                for (size_t ide=0; ide<prevDepth; ++ide)
                {
                    for (size_t wh=0; wh<mapSize; ++wh)
                    {
                        for (size_t ww=0; ww<=mapSize; ++ww)
                        {
                            auto wIdx = ode*mapSize*mapSize*depth+ide*mapSize*mapSize+wh*mapSize+ww;
                            auto iIdx = ide*prevWidth*prevHeight+(ihStart+wh)*prevHeight+(iwStart+ww);
                            val += weight[wIdx]*prevA[iIdx];
                        }
                    }
                }
                
                auto oIdx = ode*width*height+oh*width+ow;
                a[oIdx] = AFunc.f(val);
            }
        }
    }
}
//
void ConvLayer::bwdProp()
{
    calcGrad();
    
    ConvLayer* prevConvLayer = static_cast<ConvLayer*>(prevLayer);
    auto& prevDelta   = prevConvLayer->getDelta();
    const auto& prevA = prevConvLayer->getA();
    auto prevHeight   = prevConvLayer->getHeight();
    auto prevWidth    = prevConvLayer->getWidth();
    auto prevDepth    = prevConvLayer->getDepth();
    
    for (size_t ide=0; ide<prevDepth; ++ide)
    {
        for (size_t ih=0; ih<prevHeight; ++ih)
        {
            auto ohStart = ih+(mapSize-1)/2;
            
            for (size_t iw=0; iw<prevWidth; ++iw)
            {
                auto owStart = iw+(mapSize-1)/2;
                
                auto val=0.;
                for (size_t ode=0; ide<depth; ++ode)
                {
                    for (size_t wh=0; wh<mapSize; ++wh)
                    {
                        for (size_t ww=0; ww<=mapSize; ++ww)
                        {
                            auto wIdx = ode*mapSize*mapSize*depth+ide*mapSize*mapSize+wh*mapSize+ww;
                            auto oIdx = ode*prevWidth*prevHeight+(ohStart-wh)*prevHeight+(owStart-ww);
                            val += weight[wIdx]*delta[oIdx];
                        }
                    }
                }
                
                auto iIdx = ide*prevWidth*prevHeight+ih*prevWidth+iw;
                prevDelta[iIdx] = AFunc.df(prevA[iIdx])*val;
            }
        }
    }
}
//
void ConvLayer::calcGrad()
{
    ConvLayer* prevConvLayer = static_cast<ConvLayer*>(prevLayer);
    const auto& prevA = prevConvLayer->getA();
    auto prevHeight   = prevConvLayer->getHeight();
    auto prevWidth    = prevConvLayer->getWidth();
    auto prevDepth    = prevConvLayer->getDepth();
    
    for (size_t ode=0; ode<depth; ++ode)
    {
        auto valBias=0.;
        for (size_t oh=0; oh<height; ++oh)
        {
            for (size_t ow=0; ow<width; ++ow)
            {
                auto oIdx = ode*width*height+oh*width+ow;
                valBias += delta[oIdx];
            }
        }
        dbias[ode] = valBias;
        
        
        for (size_t ide=0; ide<prevDepth; ++ide)
        {
            for (size_t wh=0; wh<mapSize; ++wh)
            {
                for (size_t ww=0; ww<mapSize; ++ww)
                {
                    auto valWeight =0.;
               
                    for (size_t oh=0; oh<height; ++oh)
                    {
                        auto ihStart = oh-(mapSize-1)/2;
                        
                        for (size_t ow=0; ow<width; ++ow)
                        {
                            auto iwStart = ow-(mapSize-1)/2;

                            auto iIdx  = ide*prevWidth*prevHeight+(ihStart+wh)*prevHeight+(iwStart+ww);
                            auto oIdx  = ode*width*height+oh*width+ow;
                            valWeight +=  delta[oIdx]*prevA[iIdx];
                        }
                    }
                    
                    auto wIdx = ode*mapSize*mapSize*depth+ide*mapSize*mapSize+wh*mapSize+ww;
                    dweight[wIdx] = valWeight;
                }
            }
        }
    }
}
    
        
    
}
