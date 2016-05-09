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
{}
//
void ConvPoolLayer::validate(const Layer* prevLayer) const
{
    ConvLayer::validate(prevLayer);

    if (static_cast<const ConvLayer*>(prevLayer)->getDepth()!=depth)
        throw invalid_argument("Invalid depth");
}
//
void ConvPoolLayer::setFromPrev(const Layer* prevLayer)
{
    inputSize = prevLayer->getOutputSize();
    validate(prevLayer);
}
//
void ConvPoolLayer::resize(size_t nbData)
{
    Layer::resize(nbData);
    maxIdx.resize(outputSize*nbData);
}
//
void ConvPoolLayer::maxPool(const ConvLayer* prevCL, size_t d, size_t de, size_t ih, size_t iw, int& mIdx, real& mVal) const
{
    const auto& prevA = prevCL->getA();
    auto        cIdx  = prevCL->getIdx(d, de, ih, iw);
    
    mIdx = cIdx;
    mVal = prevA[cIdx];
    for (size_t wh=0; wh<mapSize; ++wh)
    {
        for (size_t ww=0; ww<mapSize; ++ww)
        {
            cIdx = prevCL->getIdx(d, de, ih+wh, iw+ww);
            
            if (mVal<prevA[cIdx])
            {
                mIdx = cIdx;
                mVal = prevA[cIdx];
            }
        }
    }
}
//
    
//
void ConvPoolLayer::fwdProp(const Layer* prevLayer)
{
    const ConvLayer* prevCL = static_cast<const ConvLayer*>(prevLayer);
    
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
        
                    auto oIdx = getIdx(d, de, oh, ow);
                    maxPool(prevCL, d, de, ih, iw, maxIdx[oIdx], a[oIdx]);
                    a[oIdx] = AFunc.f(a[oIdx]);
                }
            }
        }
    }
}
//
void ConvPoolLayer::bwdProp(Layer* prevLayer)
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
