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
void ConvLayer::naiveBwdProp()
{
    ConvLayer* prevConvLayer = static_cast<ConvLayer*>(prevLayer);
    auto& prevDelta       = prevConvLayer->getDelta();
    const auto& prevA     = prevConvLayer->getA();
    const auto& prevAFunc = prevConvLayer->getAFunc();
    auto prevHeight       = prevConvLayer->getHeight();
    auto prevWidth        = prevConvLayer->getWidth();
    auto prevDepth        = prevConvLayer->getDepth();
    
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
void ConvLayer::naiveCalcGrad()
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
            real valBias=0.;
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
                        real valWeight =0.;
                        for (size_t oh=0; oh<height; ++oh)
                        {
                            for (size_t ow=0; ow<width; ++ow)
                            {
                                auto iIdx  = d*prevWidth*prevHeight*prevDepth+ide*prevWidth*prevHeight+(oh+wh)*prevWidth+(ow+ww);
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
//

//
void ConvLayer::genPrevAMatFwd(size_t d, vec_r& prevAMat) const
{
    ConvLayer* prevConvLayer = static_cast<ConvLayer*>(prevLayer);
    const auto& prevA = prevConvLayer->getA();
    auto prevHeight   = prevConvLayer->getHeight();
    auto prevWidth    = prevConvLayer->getWidth();
    auto prevDepth    = prevConvLayer->getDepth();
    
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
                        auto iIdx = d*prevWidth*prevHeight*prevDepth+ide*prevWidth*prevHeight+(ih+wh)*prevWidth+(iw+ww);
                        prevAMat[prevAMatIdx++] = prevA[iIdx];
                    }
                }
            }
        }
    }
}
//
void ConvLayer::genPrevAMatGrad(size_t d, vec_r& prevAMat) const
{
    ConvLayer* prevConvLayer = static_cast<ConvLayer*>(prevLayer);
    const auto& prevA = prevConvLayer->getA();
    auto prevHeight   = prevConvLayer->getHeight();
    auto prevWidth    = prevConvLayer->getWidth();
    auto prevDepth    = prevConvLayer->getDepth();
    
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
                        
                        auto iIdx = d*prevWidth*prevHeight*prevDepth+ide*prevWidth*prevHeight+(ih+wh)*prevWidth+(iw+ww);
                        prevAMat[aIdx++] = prevA[iIdx];
                    }
                }
            }
        }
    }
}
//
void ConvLayer::genWeightMat(vec_r& weightMat) const
{
    ConvLayer* prevConvLayer = static_cast<ConvLayer*>(prevLayer);
    auto prevDepth = prevConvLayer->getDepth();
    
    auto wIdx = 0;
    for (size_t ide=0; ide<prevDepth; ++ide)
        for (size_t ode=0; ode<depth; ++ode)
            for (size_t wh=0; wh<mapSize; ++wh)
                for (size_t ww=0; ww<mapSize; ++ww)
                    weightMat[wIdx++] = weight[ode*mapSize*mapSize*prevDepth+ide*mapSize*mapSize+wh*mapSize+ww];
}
//
void ConvLayer::genIdxVec(size_t pdim, size_t dim, vector<int>& weightIdxVec) const
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
void ConvLayer::genDeltaMat(size_t d, vector<int>& hIdxVec, vector<int>& wIdxVec, vec_r& deltaMat) const
{
    ConvLayer* prevConvLayer = static_cast<ConvLayer*>(prevLayer);
    auto prevHeight       = prevConvLayer->getHeight();
    auto prevWidth        = prevConvLayer->getWidth();
    
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
                            auto oIdx = d*width*height*depth+ode*width*height+h*width+w;
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
void ConvLayer::img2MatFwdProp()
{
    if (prevLayer==nullptr)
        return;
    
    ConvLayer* prevConvLayer = static_cast<ConvLayer*>(prevLayer);
    
    auto prevDepth = prevConvLayer->getDepth();
    auto nbRow     = width*height;
    auto nbCol     = prevDepth*mapSize*mapSize;
    
    auto aIdx =0;
    for (size_t d=0; d<nbData; ++d)
    {
        vec_r ad      (nbRow*depth);
        vec_r prevAMat(nbRow*nbCol);
        
        genPrevAMatFwd(d, prevAMat);
        MatMultABt(weight, prevAMat, ad, depth, nbCol, nbRow);
        
        for (size_t ode=0; ode<depth; ++ode)
        {
            auto b = bias[ode];
            for (size_t o=0; o<nbRow; ++o)
                a[aIdx+ode*width*height+o] = AFunc.f(ad[ode*nbRow+o] + b);
        }
        
        aIdx += outputSize;
    }
}
//
void ConvLayer::img2MatBwdProp()
{
    ConvLayer* prevConvLayer = static_cast<ConvLayer*>(prevLayer);
    auto& prevDelta       = prevConvLayer->getDelta();
    const auto& prevA     = prevConvLayer->getA();
    const auto& prevAFunc = prevConvLayer->getAFunc();
    auto prevHeight       = prevConvLayer->getHeight();
    auto prevWidth        = prevConvLayer->getWidth();
    auto prevDepth        = prevConvLayer->getDepth();
    
    vec_r prevdA(prevA.size());
    for (size_t i=0; i<prevdA.size(); ++i)
        prevdA[i] = prevAFunc.df(prevA[i]);
    
    auto nbRow = prevWidth*prevHeight;
    auto nbCol = depth*mapSize*mapSize;
    
    vec_r weightMat(prevDepth*nbCol);
    genWeightMat(weightMat);
    
    vector<int> hIdxVec(mapSize*prevHeight, -1);
    vector<int> wIdxVec(mapSize*prevWidth , -1);
    genIdxVec(prevHeight, height, hIdxVec);
    genIdxVec(prevWidth , width , wIdxVec);
    
    auto dIdx =0;
    for (size_t d=0; d<nbData; ++d)
    {
        vec_r deltaMat  (nbRow*nbCol);
        vec_r prevDeltad(nbRow*prevDepth);
        
        genDeltaMat(d, hIdxVec, wIdxVec, deltaMat);
        MatMultABt(weightMat, deltaMat, prevDeltad, prevDepth, nbCol, nbRow);
        
        transform(prevDeltad.begin(), prevDeltad.end(), prevdA.begin()+dIdx, prevDelta.begin()+dIdx, [] (auto pD, auto pdA) {return pD*pdA;});
        dIdx += inputSize;
    }
}
//
void ConvLayer::img2MatCalcGrad()
{
    ConvLayer* prevConvLayer = static_cast<ConvLayer*>(prevLayer);
    auto prevDepth = prevConvLayer->getDepth();
    
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
        genPrevAMatGrad(d, prevAMat);
        MatMultABt(deltaMat, prevAMat, dWeightMat, depth, nbCol, nbRow);
        
        transform(dWeightMat.begin(), dWeightMat.end(), dweight.begin(), dweight.begin(), [] (auto dwM, auto dw) {return dw+dwM;});
        dIdx += outputSize;
    }
}



}
