//
//  ConvLayer.cpp
//  NeuralNetworks
//
//  Created by Thomas Hubert on 15/08/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#include "ConvLayer.h"
#include "LinearAlgebra.h"

namespace NN
{

ConvLayer::ConvLayer(size_t width, size_t height, size_t depth, size_t mapSize, size_t stride, const ActivationFunc& AFunc) :
    Layer(width*height*depth, 0, AFunc),
    width (width),
    height(height),
    depth (depth),
    mapSize(mapSize),
    stride(stride)
{}
//
void ConvLayer::validate(const Layer* prevLayer) const
{
    auto pLClass = prevLayer->getClass();
    if (pLClass!=LayerClass::ConvLayer && pLClass!=LayerClass::ConvPoolLayer)
        throw invalid_argument("Previous Layer of a ConvLayer must be a ConvLayer");
    
    auto prevWidth  = static_cast<const ConvLayer*>(prevLayer)->getWidth();
    auto prevHeight = static_cast<const ConvLayer*>(prevLayer)->getHeight();
    
    if (!(((prevWidth-mapSize)%stride == 0) && ((prevHeight-mapSize)%stride == 0)))
        throw invalid_argument("Invalid stride, mapSize configuration");
    if (!((width == 1+(prevWidth-mapSize)/stride) && (height== 1+(prevHeight-mapSize)/stride)))
        throw invalid_argument("Invalid size, stride, mapSize configuration");
}
//
void ConvLayer::setFromPrev(const Layer* prevLayer)
{
    inputSize = prevLayer->getOutputSize();
    
    validate(prevLayer);
    
    prevDepth  = static_cast<const ConvLayer*>(prevLayer)->getDepth();
    auto weightInputSize = mapSize*mapSize*prevDepth;
    auto weightSize = weightInputSize*depth;
    
    params.resize (depth, weightSize);
    dparams.resize(depth, weightSize);
    initParams(weightInputSize);
}
//
    
//
void ConvLayer::fwdProp(const Layer* prevLayer)
{
    const ConvLayer* prevCL = static_cast<const ConvLayer*>(prevLayer);
    const auto& prevA       = prevCL->getA();
    const auto  prevHeight  = prevCL->getHeight();
    const auto  prevWidth   = prevCL->getWidth();
    const auto& bias        = params.bias;
    const auto& weight      = params.weight;
    
    fill(a.begin(), a.end(), 0.);
    
    vector<real> weightT(prevDepth*depth*mapSize*mapSize);
    auto idx=0;
    for (size_t ide=0; ide<prevDepth; ++ide)
        for (size_t ode=0; ode<depth; ++ode)
            for (size_t wh=0; wh<mapSize; ++wh)
                for (size_t ww=0; ww<mapSize; ++ww)
                    weightT[idx++] = weight[getWIdx(ode, ide, wh, ww)];
    
    for (size_t d=0; d<nbData; ++d)
    {
        auto aStart = getIdx(d, 0, 0, 0);
        
        for (size_t ide=0; ide<prevDepth; ++ide)
        {
            auto prevAStart = prevCL->getIdx(d, ide, 0, 0);
            auto wStart     = ide*depth*mapSize*mapSize;
            CorrMat(&weightT[wStart], &prevA[prevAStart], &a[aStart], depth, mapSize, mapSize, prevHeight, prevWidth);
        }

        for (size_t ode=0; ode<depth; ++ode)
        {
            auto tmpBias = bias[ode];
            for (size_t oh=0; oh<height; ++oh)
            {
                for (size_t ow=0; ow<width; ++ow)
                {
                    auto oIdx = getIdx(d, ode, oh, ow);
                    a[oIdx] =AFunc.f(a[oIdx]+tmpBias);
                }
            }
        }
    }
}
//
void ConvLayer::bwdProp(Layer* prevLayer)
{
    ConvLayer* prevCL     = static_cast<ConvLayer*>(prevLayer);
    auto&       prevDelta = prevCL->getDelta();
    const auto& prevA     = prevCL->getA();
    const auto& prevAFunc = prevCL->getAFunc();
    const auto& weight    = params.weight;
    
    fill(prevDelta.begin(), prevDelta.end(), 0.);
    
    auto padHeight = height+2*(mapSize-1);
    auto padWidth  = width +2*(mapSize-1);
    vector<real> padDelta(depth*padHeight*padWidth);
    
    for (size_t d=0; d<nbData; ++d)
    {
        for (size_t ode=0; ode<depth; ++ode)
        {
            auto deltaStart    = getIdx(d, ode, 0, 0);
            auto padDeltaStart = ode*padHeight*padWidth+(mapSize-1)*padWidth+mapSize-1;
            
            for (size_t i=0; i<height; ++i)
                for (size_t j=0; j<width; ++j)
                    padDelta[padDeltaStart+i*padWidth+j]=delta[deltaStart+i*width+j];
        }
        
        auto prevDeltaStart = prevCL->getIdx(d, 0, 0, 0);
        for (size_t ode=0; ode<depth; ++ode)
        {
            auto padDeltaStart  = ode*padHeight*padWidth;
            auto wStart         = getWIdx(ode, 0, 0, 0);
            ConvMat(&weight[wStart], &padDelta[padDeltaStart], &prevDelta[prevDeltaStart], prevDepth, mapSize, mapSize, padHeight, padWidth);
        }
    }
    
    for (size_t i=0; i<prevDelta.size(); ++i)
        prevDelta[i] *= prevAFunc.df(prevA[i]);
}
//
void ConvLayer::calcGrad(const Layer* prevLayer)
{
    const ConvLayer* prevCL = static_cast<const ConvLayer*>(prevLayer);
    const auto& prevA       = prevCL->getA();
    const auto  prevHeight  = prevCL->getHeight();
    const auto  prevWidth   = prevCL->getWidth();
    
    dparams.reset();
    auto& dbias   = dparams.bias;
    auto& dweight = dparams.weight;
    
    auto nbCol = height*width;
    auto dBIdx = 0;
    for (size_t d=0; d<nbData; ++d)
    {
        for (size_t ode=0; ode<depth; ++ode)
        {
            dbias[ode] += accumulate(delta.begin()+dBIdx, delta.begin()+dBIdx+nbCol, 0.);
            dBIdx += nbCol;
        }
    }
    
    vector<real> dweightT(prevDepth*depth*mapSize*mapSize);
    for (size_t d=0; d<nbData; ++d)
    {
        for (size_t ide=0; ide<prevDepth; ++ide)
        {
            auto prevAStart = prevCL->getIdx(d, ide, 0, 0);
            auto deltaStart = getIdx(d, 0, 0, 0);
            auto dwStart    = ide*depth*mapSize*mapSize;
            
            CorrMat(&delta[deltaStart], &prevA[prevAStart], &dweightT[dwStart], depth, height, width, prevHeight, prevWidth);
        }
    }
    
    auto idx = 0;
    for (size_t ide=0; ide<prevDepth; ++ide)
        for (size_t ode=0; ode<depth; ++ode)
            for (size_t wh=0; wh<mapSize; ++wh)
                for (size_t ww=0; ww<mapSize; ++ww)
                    dweight[getWIdx(ode, ide, wh, ww)] = dweightT[idx++];
}


}
