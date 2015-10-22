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
    
    if (!(((prevWidth-mapSize)%stride == 0) && ((prevHeight-mapSize)%stride == 0)))
        throw "Invalid stride, mapSize configuration";
    if (!((width == 1+(prevWidth-mapSize)/stride) && (height== 1+(prevHeight-mapSize)/stride)))
        throw "Invalid size, stride, mapSize configuration";
    
    weightInputSize = mapSize*mapSize*prevDepth;
    weight.resize (weightSize);
    dweight.resize(weightSize);
    
    initParams();
}
//
void ConvLayer::fwdProp()
{
    //naiveFwdProp();
    img2MatFwdProp();
}
//
void ConvLayer::bwdProp()
{
    //naiveBwdProp();
    img2MatBwdProp();
}
//
void ConvLayer::calcGrad()
{
    //naiveCalcGrad();
    img2MatCalcGrad();
}
//
void ConvLayer::naiveFwdProp()
{
    if (prevLayer==nullptr)
        return;
    
    ConvLayer* prevCL = static_cast<ConvLayer*>(prevLayer);
    const auto& prevA = prevCL->getA();
    auto prevDepth    = prevCL->getDepth();
    
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
void ConvLayer::naiveBwdProp()
{
    ConvLayer* prevCL = static_cast<ConvLayer*>(prevLayer);
    auto& prevDelta       = prevCL->getDelta();
    const auto& prevA     = prevCL->getA();
    const auto& prevAFunc = prevCL->getAFunc();
    auto prevDepth        = prevCL->getDepth();
    
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
void ConvLayer::naiveCalcGrad()
{
    ConvLayer* prevCL = static_cast<ConvLayer*>(prevLayer);
    const auto& prevA = prevCL->getA();
    auto prevDepth    = prevCL->getDepth();
    
    fill(dbias.begin()  , dbias.end()  , 0.);
    fill(dweight.begin(), dweight.end(), 0.);
    
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
void ConvLayer::genFwdPrevAMat(size_t d, vec_r& prevAMat) const
{
    ConvLayer* prevCL = static_cast<ConvLayer*>(prevLayer);
    const auto& prevA = prevCL->getA();
    auto prevDepth    = prevCL->getDepth();
    
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
    ConvLayer* prevCL = static_cast<ConvLayer*>(prevLayer);
    auto prevDepth = prevCL->getDepth();
    
    auto wIdx = 0;
    for (size_t ide=0; ide<prevDepth; ++ide)
        for (size_t ode=0; ode<depth; ++ode)
            for (size_t wh=0; wh<mapSize; ++wh)
                for (size_t ww=0; ww<mapSize; ++ww)
                    weightMat[wIdx++] = weight[getWIdx(ode, ide, wh, ww)];
}
//
void ConvLayer::genBwdDeltaMat(size_t d, vec_i& hIdxVec, vec_i& wIdxVec, vec_r& deltaMat) const
{
    ConvLayer* prevCL = static_cast<ConvLayer*>(prevLayer);
    auto prevHeight       = prevCL->getHeight();
    auto prevWidth        = prevCL->getWidth();
    
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
void ConvLayer::genGradPrevAMat(size_t d, vec_r& prevAMat) const
{
    ConvLayer* prevCL = static_cast<ConvLayer*>(prevLayer);
    const auto& prevA = prevCL->getA();
    auto prevDepth    = prevCL->getDepth();
    
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
void ConvLayer::img2MatFwdProp()
{
    if (prevLayer==nullptr)
        return;
    
    ConvLayer* prevCL = static_cast<ConvLayer*>(prevLayer);
    
    auto prevDepth = prevCL->getDepth();
    auto nbRow     = width*height;
    auto nbCol     = prevDepth*mapSize*mapSize;
    
    auto aIdx =0;
    for (size_t d=0; d<nbData; ++d)
    {
        vec_r ad      (nbRow*depth);
        vec_r prevAMat(nbRow*nbCol);
        
        genFwdPrevAMat(d, prevAMat);
        MatMultABt(weight, prevAMat, ad, depth, nbCol, nbRow);
        
        for (size_t ode=0; ode<depth; ++ode)
        {
            auto b = bias[ode];
            for (size_t o=0; o<nbRow; ++o)
                a[aIdx++] = AFunc.f(ad[ode*nbRow+o] + b);
        }
    }
}
//
void ConvLayer::img2MatBwdProp()
{
    ConvLayer* prevCL = static_cast<ConvLayer*>(prevLayer);
    auto& prevDelta       = prevCL->getDelta();
    const auto& prevA     = prevCL->getA();
    const auto& prevAFunc = prevCL->getAFunc();
    auto prevHeight       = prevCL->getHeight();
    auto prevWidth        = prevCL->getWidth();
    auto prevDepth        = prevCL->getDepth();
    
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
        
        genBwdDeltaMat(d, hIdxVec, wIdxVec, deltaMat);
        MatMultABt(weightMat, deltaMat, prevDeltad, prevDepth, nbCol, nbRow);
        
        transform(prevDeltad.begin(), prevDeltad.end(), prevdA.begin()+dIdx, prevDelta.begin()+dIdx, [] (auto pD, auto pdA) {return pD*pdA;});
        dIdx += inputSize;
    }
}
//
void ConvLayer::img2MatCalcGrad()
{
    ConvLayer* prevCL = static_cast<ConvLayer*>(prevLayer);
    auto prevDepth = prevCL->getDepth();
    
    auto nbRow = prevDepth*mapSize*mapSize;
    auto nbCol = height*width;
    
    fill(dbias.begin()  , dbias.end()  , 0.);
    fill(dweight.begin(), dweight.end(), 0.);
    
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
        vec_r deltaMat  (depth*nbCol);
        
        copy(delta.begin()+dIdx, delta.begin()+dIdx+outputSize, deltaMat.begin());
        genGradPrevAMat(d, prevAMat);
        MatMultABt(deltaMat, prevAMat, dWeightMat, depth, nbCol, nbRow);
        
        transform(dWeightMat.begin(), dWeightMat.end(), dweight.begin(), dweight.begin(), [] (auto dwM, auto dw) {return dw+dwM;});
        dIdx += outputSize;
    }
}



}
