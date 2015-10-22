//
//  ConvPoolLayer.cpp
//  NeuralNetworks
//
//  Created by Thomas Hubert on 05/09/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#include "ConvPoolLayer.h"

namespace NN
{
    
ConvPoolLayer::ConvPoolLayer(size_t width, size_t height, size_t depth, size_t mapSize, size_t stride, const ActivationFunc& AFunc) :
    ConvLayer(width, height, depth, mapSize, stride, AFunc)
{
    bias.resize(0);
    dbias.resize(0);
    weight.resize (0);
    dweight.resize(0);
}
//
void ConvPoolLayer::setPrevLayer(Layer* prev)
{
    prevLayer = prev;
    inputSize = prevLayer->getOutputSize();
    weightInputSize = 1.;
    
    assert(prevLayer->getClass() == LayerClass::ConvLayer || prevLayer->getClass() == LayerClass::ConvPoolLayer);
    
    auto prevWidth  = static_cast<ConvLayer*>(prevLayer)->getWidth();
    auto prevHeight = static_cast<ConvLayer*>(prevLayer)->getHeight();
    auto prevDepth  = static_cast<ConvLayer*>(prevLayer)->getDepth();
    
    if (!(((prevWidth-mapSize)%stride == 0) && ((prevHeight-mapSize)%stride == 0)))
        throw "Invalid stride, mapSize configuration";
    if (!((width == 1+(prevWidth-mapSize)/stride) && (height== 1+(prevHeight-mapSize)/stride)))
        throw "Invalid size, stride, mapSize configuration";
    if (prevDepth!=depth)
        throw "Invalid depth";
    
    initParams();
}
//
void ConvPoolLayer::resize(size_t nbData)
{
    Layer::resize(nbData);
    maxIdx.resize(outputSize*nbData);
}
//
void ConvPoolLayer::fwdProp()
{
    ConvLayer* prevCL = static_cast<ConvLayer*>(prevLayer);
    const auto& prevA = prevCL->getA();
    
    for (size_t d=0; d<nbData; ++d)
    {
        for (size_t de=0; de<depth; ++de)
        {
            for (size_t oh=0; oh<height; ++oh)
            {
                auto ih = oh*stride;
                
                for (size_t ow=0; ow<width; ++ow)
                {
                    auto iw = ow*stride;
                    
                    auto iIdx = prevCL->getIdx(d, de, ih, iw);
                    auto mIdx = iIdx;
                    auto val  = prevA[iIdx];
                    for (size_t wh=0; wh<mapSize; ++wh)
                    {
                        for (size_t ww=0; ww<mapSize; ++ww)
                        {
                            iIdx = prevCL->getIdx(d, de, ih+wh, iw+ww);
                            if (val<prevA[iIdx])
                            {
                                mIdx = iIdx;
                                val = prevA[iIdx];
                            }
                        }
                    }
                    
                    auto oIdx    = getIdx(d, de, oh, ow);
                    a[oIdx]      = AFunc.f(val);
                    maxIdx[oIdx] = mIdx;
                }
            }
        }
    }
}
//
void ConvPoolLayer::bwdProp()
{
    ConvLayer*  prevCL    = static_cast<ConvLayer*>(prevLayer);
    auto&       prevDelta = prevCL->getDelta();
    const auto& prevA     = prevCL->getA();
    const auto& prevAFunc = prevCL->getAFunc();

    fill(prevDelta.begin(), prevDelta.end(), 0.);
    
    for (size_t d=0; d<nbData; ++d)
    {
        for (size_t de=0; de<depth; ++de)
        {
            for (size_t oh=0; oh<height; ++oh)
            {
                for (size_t ow=0; ow<width; ++ow)
                {
                    auto oIdx = getIdx(d, de, oh, ow);
                    auto iIdx = maxIdx[oIdx];
                    
                    prevDelta[iIdx] += delta[oIdx] * prevAFunc.df(prevA[iIdx]);
                }
            }
        }
    }
}
//
    
    
    
}
