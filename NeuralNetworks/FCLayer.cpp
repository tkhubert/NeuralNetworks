//
//  FCLayer.cpp
//  NeuralNetworks
//
//  Created by Thomas Hubert on 04/07/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#include "FCLayer.h"
#include "LinearAlgebra.h"

namespace NN {
    
//
FCLayer::FCLayer(size_t size, real dropRate, const ActivationFunc& AFunc) :
    Layer(size, dropRate, AFunc)
{}
//
void FCLayer::setPrevLayer(Layer* prev)
{
    prevLayer = prev;
    inputSize = prevLayer->getOutputSize();
    
    params.resize (outputSize, inputSize*outputSize, inputSize);
    dparams.resize(outputSize, inputSize*outputSize, inputSize);
    params.initParams(gen);
}
//
void FCLayer::fwdProp()
{
    // A_(l+1)  = AFunc( A_l W^T_(l+1)  + B_(l+1))
    if (prevLayer==nullptr)
    {
        transform(a.begin(), a.end(), drop.begin(), a.begin(), [] (auto a, auto d) {return a*d;});
        return;
    }
    
    const auto& prevA  = prevLayer->getA();
    const auto  bias   = params.getCBPtr();
    const auto  weight = params.getCWPtr();
    
    MatMultABt(&prevA[0], weight, &a[0], nbData, inputSize, outputSize);
    
    for (size_t d=0; d<nbData; ++d)
        for (size_t o=0; o<outputSize; ++o)
            a[d*outputSize+o]  = AFunc.f(a[d*outputSize+o]+bias[o])*drop[d*outputSize+o];
}
//
void FCLayer::bwdProp()
{
    // D_l = (D(l+1) W_(l+1)).dA_l
    const auto& prevA     = prevLayer->getA();
    const auto& prevDrop  = prevLayer->getDrop();
    const auto& prevAFunc = prevLayer->getAFunc();
    auto& prevDelta       = prevLayer->getDelta();
    const auto  weight    = params.getCWPtr();
    
    MatMultAB(&delta[0], weight, &prevDelta[0], nbData, outputSize, inputSize);
    
    for (size_t i=0; i<prevDelta.size(); ++i)
        prevDelta[i] *= prevAFunc.df(prevA[i])*prevDrop[i];
}
//
void FCLayer::calcGrad()
{
    dparams.reset();
    auto dbias   = dparams.getBPtr();
    auto dweight = dparams.getWPtr();
    
    const auto& prevA = prevLayer->getA();
    
    for (size_t d=0; d<nbData; ++d)
    {
        for (size_t o=0; o<outputSize; ++o)
        {
            auto tmpDelta = delta[d*outputSize+o];
            
            dbias[o] += tmpDelta;
            for (size_t i=0; i<inputSize; ++i)
                dweight[o*inputSize+i] += tmpDelta * prevA[d*inputSize+i];
        }
    }    
}
//
}
