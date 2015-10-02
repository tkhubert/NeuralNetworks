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
};
//
void ConvLayer::setPrevLayer(Layer* prev)
{
    prevLayer = prev;
    inputSize = prevLayer->getOutputSize();
    
    assert(prevLayer->getClass() == LayerClass::ConvLayer || prevLayer->getClass() == LayerClass::ConvPoolLayer);
    auto prevDepth  = static_cast<ConvLayer*>(prevLayer)->getDepth();
    auto weightSize = mapSize*mapSize*depth*prevDepth;
    
    weightInputSize = mapSize*mapSize*prevDepth;
    weight.resize (weightSize);
    dweight.resize(weightSize);
    
    initParams();
}
//
void ConvLayer::fwdProp()
{
    if (prevLayer==nullptr)
        return;
    
    ConvLayer* prevConvLayer = static_cast<ConvLayer*>(prevLayer);
    const auto& prevA = prevConvLayer->getA();
    auto prevHeight   = prevConvLayer->getHeight();
    auto prevWidth    = prevConvLayer->getWidth();
    auto prevDepth    = prevConvLayer->getDepth();
    
    for (size_t d=0; d<nbData; ++d)
    {
        for (size_t ode=0; ode<depth; ++ode)
        {
            for (size_t oh=0; oh<height; ++oh)
            {
                for (size_t ow=0; ow<width; ++ow)
                {
                    auto val = bias[ode];
                    for (size_t ide=0; ide<prevDepth; ++ide)
                    {
                        for (size_t wh=0; wh<mapSize; ++wh)
                        {
                            for (size_t ww=0; ww<mapSize; ++ww)
                            {
                                auto wIdx = ode*mapSize*mapSize*prevDepth+ide*mapSize*mapSize+wh*mapSize+ww;
                                auto iIdx = d*prevWidth*prevHeight*prevDepth+ide*prevWidth*prevHeight+(oh+wh)*prevWidth+(ow+ww);
                                val += weight[wIdx]*prevA[iIdx];
                            }
                        }
                    }
                    
                    auto oIdx = d*width*height*depth+ode*width*height+oh*width+ow;
                    a[oIdx] = AFunc.f(val);
                }
            }
        }
    }
}
//
void ConvLayer::bwdProp()
{
    ConvLayer* prevConvLayer = static_cast<ConvLayer*>(prevLayer);
    auto& prevDelta       = prevConvLayer->getDelta();
    const auto& prevA     = prevConvLayer->getA();
    const auto& prevAFunc = prevConvLayer->getAFunc();
    auto prevHeight       = prevConvLayer->getHeight();
    auto prevWidth        = prevConvLayer->getWidth();
    auto prevDepth        = prevConvLayer->getDepth();
    
    for (size_t d=0; d<nbData; ++d)
    {
        for (size_t ide=0; ide<prevDepth; ++ide)
        {
            for (size_t ih=0; ih<prevHeight; ++ih)
            {
                auto whs = max<size_t>(0, ih-height+1);
                auto whe = min(mapSize,ih+1);
                
                for (size_t iw=0; iw<prevWidth; ++iw)
                {
                    auto wws = max<size_t>(0, iw-width+1);
                    auto wwe = min(mapSize,iw+1);
                    
                    float val=0.;
                    for (size_t ode=0; ode<depth; ++ode)
                    {
                        for (size_t wh=whs; wh<whe; ++wh)
                        {
                            for (size_t ww=wws; ww<wwe; ++ww)
                            {
                                auto wIdx = ode*mapSize*mapSize*prevDepth+ide*mapSize*mapSize+wh*mapSize+ww;
                                auto oIdx = d*width*height*depth+ode*width*height+(ih-wh)*width+(iw-ww);
                                val += weight[wIdx]*delta[oIdx];
                            }
                        }
                    }
                    
                    auto iIdx = d*prevWidth*prevHeight*prevDepth+ide*prevWidth*prevHeight+ih*prevWidth+iw;
                    prevDelta[iIdx] = prevAFunc.df(prevA[iIdx])*val;
                }
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
    
    for (size_t d=0; d<nbData; ++d)
    {
        for (size_t ode=0; ode<depth; ++ode)
        {
            float valBias=0.;
            for (size_t oh=0; oh<height; ++oh)
            {
                for (size_t ow=0; ow<width; ++ow)
                {
                    auto oIdx = d*width*height*depth+ode*width*height+oh*width+ow;
                    valBias += delta[oIdx];
                }
            }
            dbias[ode] += valBias;
            
            
            for (size_t ide=0; ide<prevDepth; ++ide)
            {
                for (size_t wh=0; wh<mapSize; ++wh)
                {
                    for (size_t ww=0; ww<mapSize; ++ww)
                    {
                        float valWeight =0.;
                        for (size_t oh=0; oh<height; ++oh)
                        {
                            for (size_t ow=0; ow<width; ++ow)
                            {
                                auto iIdx  = d*prevWidth*prevHeight*prevDepth+ide*prevWidth*prevHeight+(oh+wh)*prevHeight+(ow+ww);
                                auto oIdx  = d*width*height*depth+ode*width*height+oh*width+ow;
                                valWeight +=  delta[oIdx]*prevA[iIdx];
                            }
                        }
                        
                        auto wIdx = ode*mapSize*mapSize*prevDepth+ide*mapSize*mapSize+wh*mapSize+ww;
                        dweight[wIdx] += valWeight;
                    }
                }
            }
        }
    }
}
    
        
    
}
