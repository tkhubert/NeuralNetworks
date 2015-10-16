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
    Layer(width*height*depth, 0, AFunc),
    width (width),
    height(height),
    depth (depth),
    mapSize(mapSize),
    stride(stride)
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
    
    auto prevWidth  = static_cast<ConvLayer*>(prevLayer)->getWidth();
    auto prevHeight = static_cast<ConvLayer*>(prevLayer)->getHeight();
    auto prevDepth  = static_cast<ConvLayer*>(prevLayer)->getDepth();
    auto weightSize = mapSize*mapSize*depth*prevDepth;
    
    assert((prevWidth -mapSize) % stride == 0);
    assert((prevHeight-mapSize) % stride == 0);
    assert(width == 1+(prevWidth -mapSize)/stride);
    assert(height== 1+(prevHeight-mapSize)/stride);
    
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
                auto ih = oh*stride;
                
                for (size_t ow=0; ow<width; ++ow)
                {
                    auto iw = ow*stride;

                    auto val = bias[ode];
                    for (size_t ide=0; ide<prevDepth; ++ide)
                    {
                        for (size_t wh=0; wh<mapSize; ++wh)
                        {
                            for (size_t ww=0; ww<mapSize; ++ww)
                            {
                                auto wIdx = ode*mapSize*mapSize*prevDepth+ide*mapSize*mapSize+wh*mapSize+ww;
                                auto iIdx = d*prevWidth*prevHeight*prevDepth+ide*prevWidth*prevHeight+(ih+wh)*prevWidth+(iw+ww);
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
    
    fill(prevDelta.begin(), prevDelta.end(), 0.);
    
    vector<float> prevdA(prevA.size());
    for (size_t i=0; i<prevdA.size(); ++i)
        prevdA[i] = prevAFunc.df(prevA[i]);
    
    for (size_t d=0; d<nbData; ++d)
    {
        for (size_t ode=0; ode<depth; ++ode)
        {
            for (size_t oh=0; oh<height; ++oh)
            {
                auto ih = oh*stride;
                for (size_t ow=0; ow<width; ++ow)
                {
                    auto iw = ow*stride;
                    auto oIdx = d*width*height*depth+ode*width*height+oh*width+ow;
                    
                    for (size_t ide=0; ide<prevDepth; ++ide)
                    {
                        for (size_t wh=0; wh<mapSize; ++wh)
                        {
                            for (size_t ww=0; ww<mapSize; ++ww)
                            {
                                auto wIdx    = ode*mapSize*mapSize*prevDepth+ide*mapSize*mapSize+wh*mapSize+ww;
                                auto iIdx    = d*prevWidth*prevHeight*prevDepth+ide*prevWidth*prevHeight+(ih+wh)*prevWidth+(iw+ww);
                                prevDelta[iIdx] += delta[oIdx] * weight[wIdx] * prevdA[iIdx];
                            }
                        }
                    }
                    
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
