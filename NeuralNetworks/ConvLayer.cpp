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
void ConvLayer::fwdProp(const Layer* prevLayer)
{
    //naiveFwdProp(prevLayer);
    img2MatFwdProp(prevLayer);
}
//
void ConvLayer::bwdProp(Layer* prevLayer)
{
    //naiveBwdProp(prevLayer);
    img2MatBwdProp(prevLayer);
}
//
void ConvLayer::calcGrad(const Layer* prevLayer)
{
    //naiveCalcGrad(prevLayer);
    img2MatCalcGrad(prevLayer);
}
//
void ConvLayer::naiveFwdProp(const Layer* prevLayer)
{
    const ConvLayer* prevCL  = static_cast<const ConvLayer*>(prevLayer);
    const auto& prevA  = prevCL->getA();
    const auto& bias   = params.bias;
    const auto& weight = params.weight;
    
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
                                auto wIdx = getWIdx(ode, ide, wh, ww);
                                auto iIdx = prevCL->getIdx(d, ide, ih+wh, iw+ww);
                                val += weight[wIdx]*prevA[iIdx];
                            }
                        }
                    }
                    
                    auto oIdx = getIdx(d, ode, oh, ow);
                    a[oIdx] = AFunc.f(val);
                }
            }
        }
    }
}
//
void ConvLayer::naiveBwdProp(Layer* prevLayer)
{
    ConvLayer* prevCL = static_cast<ConvLayer*>(prevLayer);
    auto& prevDelta       = prevCL->getDelta();
    const auto& prevA     = prevCL->getA();
    const auto& prevAFunc = prevCL->getAFunc();
    const auto& weight    = params.weight;
    
    fill(prevDelta.begin(), prevDelta.end(), 0.);
    
    vec_r prevdA(prevA.size());
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
                    auto oIdx = getIdx(d, ode, oh, ow);
                    
                    for (size_t ide=0; ide<prevDepth; ++ide)
                    {
                        for (size_t wh=0; wh<mapSize; ++wh)
                        {
                            for (size_t ww=0; ww<mapSize; ++ww)
                            {
                                auto wIdx    = getWIdx(ode, ide, wh, ww);
                                auto iIdx    = prevCL->getIdx(d, ide, ih+wh, iw+ww);
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
void ConvLayer::naiveCalcGrad(const Layer* prevLayer)
{
    const ConvLayer* prevCL = static_cast<const ConvLayer*>(prevLayer);
    const auto& prevA = prevCL->getA();
    
    dparams.reset();
    auto& dbias   = dparams.bias;
    auto& dweight = dparams.weight;
    
    for (size_t d=0; d<nbData; ++d)
    {
        for (size_t ode=0; ode<depth; ++ode)
        {
            real valBias=0.;
            for (size_t oh=0; oh<height; ++oh)
            {
                for (size_t ow=0; ow<width; ++ow)
                {
                    auto oIdx = getIdx(d, ode, oh, ow);
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
                        real valWeight =0.;
                        for (size_t oh=0; oh<height; ++oh)
                        {
                            for (size_t ow=0; ow<width; ++ow)
                            {
                                auto iIdx  = prevCL->getIdx(d, ide, oh+wh, ow+ww);
                                auto oIdx  = getIdx(d, ode, oh, ow);
                                valWeight +=  delta[oIdx]*prevA[iIdx];
                            }
                        }
                        
                        auto wIdx = getWIdx(ode, ide, wh, ww);
                        dweight[wIdx] += valWeight;
                    }
                }
            }
        }
    }
}
//

//
void ConvLayer::genFwdPrevAMat(size_t d, vec_r& prevAMat, const Layer* prevLayer) const
{
    const ConvLayer* prevCL = static_cast<const ConvLayer*>(prevLayer);
    const auto& prevA = prevCL->getA();
    
    auto prevAMatIdx = 0;
    for (size_t oh=0; oh<height; ++oh)
    {
        auto ih = oh*stride;
        
        for (size_t ow=0; ow<width; ++ow)
        {
            auto iw = ow*stride;
            
            for (size_t ide=0; ide<prevDepth; ++ide)
            {
                for (size_t wh=0; wh<mapSize; ++wh)
                {
                    for (size_t ww=0; ww<mapSize; ++ww)
                    {
                        auto iIdx = prevCL->getIdx(d, ide, ih+wh, iw+ww);
                        prevAMat[prevAMatIdx++] = prevA[iIdx];
                    }
                }
            }
        }
    }
}
//
void ConvLayer::genBwdIdxVec(size_t pdim, size_t dim, vec_i& weightIdxVec) const
{
    auto wIdx = 0;
    for (int i=0; i<pdim; ++i)
    {
        for (int w=0; w<mapSize; ++w)
        {
            int h = i-w;
            if (h>=0 && (h%stride==0) && h<dim*stride)
                weightIdxVec[wIdx] = h/stride;
            ++wIdx;
        }
    }
}
//
void ConvLayer::genBwdWeightMat(vec_r& weightMat) const
{
    const auto& weight = params.weight;
    
    auto wIdx = 0;
    for (size_t ide=0; ide<prevDepth; ++ide)
        for (size_t ode=0; ode<depth; ++ode)
            for (size_t wh=0; wh<mapSize; ++wh)
                for (size_t ww=0; ww<mapSize; ++ww)
                    weightMat[wIdx++] = weight[getWIdx(ode, ide, wh, ww)];
}
//
void ConvLayer::genBwdDeltaMat(size_t d, vec_i& hIdxVec, vec_i& wIdxVec, vec_r& deltaMat, const Layer* prevLayer) const
{
    const ConvLayer* prevCL = static_cast<const ConvLayer*>(prevLayer);
    auto prevHeight   = prevCL->getHeight();
    auto prevWidth    = prevCL->getWidth();
    
    auto deltaMatIdx = 0;
    for (size_t ih=0; ih<prevHeight; ++ih)
    {
        for (size_t iw=0; iw<prevWidth; ++iw)
        {
            for (size_t ode=0; ode<depth; ++ode)
            {
                for (size_t wh=0; wh<mapSize; ++wh)
                {
                    auto h = hIdxVec[ih*mapSize+wh];
                    
                    for (size_t ww=0; ww<mapSize; ++ww)
                    {
                        auto w = wIdxVec[iw*mapSize+ww];
                        
                        if (h>=0 && w>=0)
                        {
                            auto oIdx = getIdx(d, ode, h, w);
                            deltaMat[deltaMatIdx] = delta[oIdx];
                        }
                        ++deltaMatIdx;
                    }
                }
            }
        }
    }
}
//
void ConvLayer::genGradPrevAMat(size_t d, vec_r& prevAMat, const Layer* prevLayer) const
{
    const ConvLayer* prevCL = static_cast<const ConvLayer*>(prevLayer);
    const auto& prevA = prevCL->getA();
    
    auto aIdx = 0;
    for (size_t ide=0; ide<prevDepth; ++ide)
    {
        for (size_t wh=0; wh<mapSize; ++wh)
        {
            for (size_t ww=0; ww<mapSize; ++ww)
            {
                for (size_t oh=0; oh<height; ++oh)
                {
                    auto ih = oh*stride;
                    for (size_t ow=0; ow<width; ++ow)
                    {
                        auto iw = ow*stride;
                        
                        auto iIdx = prevCL->getIdx(d, ide, ih+wh, iw+ww);
                        prevAMat[aIdx++] = prevA[iIdx];
                    }
                }
            }
        }
    }
}
//
void ConvLayer::img2MatFwdProp(const Layer* prevLayer)
{   
    auto nbRow         = width*height;
    auto nbCol         = prevDepth*mapSize*mapSize;
    const auto& bias   = params.bias;
    const auto& weight = params.weight;
    
    auto aIdx =0;
    for (size_t d=0; d<nbData; ++d)
    {
        vec_r ad      (nbRow*depth);
        vec_r prevAMat(nbRow*nbCol);
        
        genFwdPrevAMat(d, prevAMat, prevLayer);
        MatMultABt(&weight[0], &prevAMat[0], &ad[0], depth, nbCol, nbRow);
        
        for (size_t ode=0; ode<depth; ++ode)
        {
            auto b = bias[ode];
            for (size_t o=0; o<nbRow; ++o)
                a[aIdx++] = AFunc.f(ad[ode*nbRow+o] + b);
        }
    }
}
//
void ConvLayer::img2MatBwdProp(Layer* prevLayer)
{
    ConvLayer* prevCL = static_cast<ConvLayer*>(prevLayer);
    auto& prevDelta       = prevCL->getDelta();
    const auto& prevA     = prevCL->getA();
    const auto& prevAFunc = prevCL->getAFunc();
    auto prevHeight       = prevCL->getHeight();
    auto prevWidth        = prevCL->getWidth();
    
    vec_r prevdA(prevA.size());
    for (size_t i=0; i<prevdA.size(); ++i)
        prevdA[i] = prevAFunc.df(prevA[i]);
    
    auto nbRow = prevWidth*prevHeight;
    auto nbCol = depth*mapSize*mapSize;
    
    vec_r weightMat(prevDepth*nbCol);
    genBwdWeightMat(weightMat);
    
    vec_i hIdxVec(mapSize*prevHeight, -1);
    vec_i wIdxVec(mapSize*prevWidth , -1);
    genBwdIdxVec(prevHeight, height, hIdxVec);
    genBwdIdxVec(prevWidth , width , wIdxVec);
    
    auto dIdx =0;
    for (size_t d=0; d<nbData; ++d)
    {
        vec_r deltaMat  (nbRow*nbCol);
        vec_r prevDeltad(nbRow*prevDepth);
        
        genBwdDeltaMat(d, hIdxVec, wIdxVec, deltaMat, prevLayer);
        MatMultABt(&weightMat[0], &deltaMat[0], &prevDeltad[0], prevDepth, nbCol, nbRow);
        
        transform(prevDeltad.begin(), prevDeltad.end(), prevdA.begin()+dIdx, prevDelta.begin()+dIdx, [] (auto pD, auto pdA) {return pD*pdA;});
        dIdx += inputSize;
    }
}
//
void ConvLayer::img2MatCalcGrad(const Layer* prevLayer)
{
    auto nbRow = prevDepth*mapSize*mapSize;
    auto nbCol = height*width;
    
    dparams.reset();
    auto& dbias   = dparams.bias;
    auto& dweight = dparams.weight;
    
    auto dIdx  = 0;
    auto dBIdx = 0;
    for (size_t d=0; d<nbData; ++d)
    {
        for (size_t ode=0; ode<depth; ++ode)
        {
            dbias[ode] += accumulate(delta.begin()+dBIdx, delta.begin()+dBIdx+nbCol, 0.);
            dBIdx += nbCol;
        }
        
        vec_r prevAMat  (nbRow*nbCol);
        vec_r dWeightMat(depth*nbRow);
        
        genGradPrevAMat(d, prevAMat, prevLayer);
        MatMultABt(&delta[dIdx], &prevAMat[0], &dWeightMat[0], depth, nbCol, nbRow);
        
        transform(dWeightMat.begin(), dWeightMat.end(), dweight.begin(), dweight.begin(), [] (auto dwM, auto dw) {return dw+dwM;});
        dIdx += outputSize;
    }
}



}
