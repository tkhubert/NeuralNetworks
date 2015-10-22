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
{
    auto biasSize = outputSize;
    bias.resize (biasSize);
    dbias.resize(biasSize);
}
//
void FCLayer::setPrevLayer(Layer* prev)
{
    prevLayer = prev;
    inputSize = prevLayer->getOutputSize();
    
    auto weightSize = inputSize*outputSize;
    weightInputSize = inputSize;
    weight.resize (weightSize);
    dweight.resize(weightSize);
    
    initParams();
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
    
    const auto& prevA = prevLayer->getA();
    
    MatMultABt(prevA, weight, a, nbData, inputSize, outputSize);
    
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
    
    MatMultAB(delta, weight, prevDelta, nbData, outputSize, inputSize);
    
    for (size_t i=0; i<prevDelta.size(); ++i)
        prevDelta[i] *= prevAFunc.df(prevA[i])*prevDrop[i];
}
//
void FCLayer::calcGrad()
{
    fill(dbias.begin()  , dbias.end()  , 0.);
    fill(dweight.begin(), dweight.end(), 0.);
    
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
