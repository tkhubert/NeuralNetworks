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
    vbias.resize(0);
    weight.resize (0);
    dweight.resize(0);
    vweight.resize(0);
}
//
void ConvPoolLayer::setPrevLayer(Layer* prev)
{
    prevLayer = prev;
    inputSize = prevLayer->getOutputSize();
    
    assert(prevLayer->getClass() == LayerClass::ConvLayer || prevLayer->getClass() == LayerClass::ConvPoolLayer);
    assert(static_cast<ConvLayer*>(prevLayer)->getDepth() == depth);
    initParams();
}
//
void ConvPoolLayer::fwdProp()
{
    ConvLayer* prevConvLayer = static_cast<ConvLayer*>(prevLayer);
    const auto& prevA = prevConvLayer->getA();
    auto prevHeight   = prevConvLayer->getHeight();
    auto prevWidth    = prevConvLayer->getWidth();
    auto prevDepth    = prevConvLayer->getDepth();
    
    for (size_t d=0; d<nbData; ++d)
    {
        for (size_t de=0; de<depth; ++de)
        {
            for (size_t oh=0; oh<height; ++oh)
            {
                auto ihStart = mapSize*oh;
                
                for (size_t ow=0; ow<width; ++ow)
                {
                    auto iwStart = mapSize*ow;
                    
                    auto iIdx = d*prevWidth*prevHeight*prevDepth+de*prevWidth*prevHeight+ihStart*prevHeight+iwStart;
                    auto mIdx = iIdx;
                    auto val  = prevA[iIdx];
                    for (size_t wh=0; wh<mapSize; ++wh)
                    {
                        for (size_t ww=0; ww<mapSize; ++ww)
                        {
                            iIdx = d*prevWidth*prevHeight*prevDepth+de*prevWidth*prevHeight+(ihStart+wh)*prevHeight+(iwStart+ww);
                            if (val<prevA[iIdx])
                            {
                                val = prevA[iIdx];
                                mIdx = iIdx;
                            }
                        }
                    }
                    
                    auto oIdx = d*width*height*depth+de*width*height+oh*width+ow;
                    a[oIdx] = val;
                    
                    maxIdx.emplace(mIdx, oIdx);
                }
            }
        }
    }
}
//
void ConvPoolLayer::bwdProp()
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
        for (size_t de=0; de<prevDepth; ++de)
        {
            for (size_t ih=0; ih<prevHeight; ++ih)
            {
                for (size_t iw=0; iw<prevWidth; ++iw)
                {
                    auto iIdx = d*prevWidth*prevHeight*prevDepth+de*prevWidth*prevHeight+ih*prevWidth+iw;
                    
                    auto val = 0.;
                    auto range = maxIdx.equal_range(iIdx);
                    for (auto itr=range.first; itr!=range.second; ++itr)
                    {
                        auto oIdx = itr->second;
                        val += delta[oIdx] * prevAFunc.df(prevA[iIdx]);
                    }
                    
                    prevDelta[iIdx] = val;
                }
            }
        }
    }
}
//
    
    
    
}
