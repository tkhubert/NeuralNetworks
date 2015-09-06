//
//  FCLayer.cpp
//  NeuralNetworks
//
//  Created by Thomas Hubert on 04/07/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#include "FCLayer.h"

namespace NN {
    
//
FCLayer::FCLayer(size_t size, float dropRate, const ActivationFunc& AFunc) :
    Layer(size, dropRate, AFunc)
{
    auto biasSize = outputSize;
    bias.resize (biasSize);
    dbias.resize(biasSize);
    vbias.resize(biasSize);
}
//
void FCLayer::setPrevLayer(Layer* prev)
{
    prevLayer = prev;
    inputSize = prevLayer->getOutputSize();
    
    auto weightSize = inputSize*outputSize;
    weight.resize (weightSize);
    dweight.resize(weightSize);
    vweight.resize(weightSize);
    
    initParams();
}
//
void FCLayer::fwdProp()
{
    if (prevLayer==nullptr)
    {
        transform(a.begin(), a.end(), drop.begin(), a.begin(), [] (auto a, auto d) {return a*d;});
        return;
    }
    
    // A_(l+1)  = AFunc( W_(l+1) A_l + B_(l+1))
    const auto& prevA = prevLayer->getA();
    
    for (size_t d=0; d<nbData; ++d)
    {
        for (size_t o=0; o<outputSize; ++o)
        {
            auto val=bias[o];
            for (size_t i=0; i<inputSize; ++i)
                val+= weight[o*inputSize+i]*prevA[d*inputSize+i];
            
            a[d*outputSize+o]  = AFunc.f(val)*drop[d*outputSize+o];
        }
    }
}
//
void FCLayer::bwdProp()
{
    // D_l = (W'_(l+1) D(l+1)) . dA_l
    const auto& prevA     = prevLayer->getA();
    const auto& prevDrop  = prevLayer->getDrop();
    const auto& prevAFunc = prevLayer->getAFunc();
    auto& prevDelta       = prevLayer->getDelta();
    
    float tmp[inputSize][outputSize];
    for (size_t i=0; i<inputSize; ++i)
        for (size_t o=0; o<outputSize; ++o)
            tmp[i][o] = weight[o*inputSize+i];
    
    for (size_t d=0; d<nbData; ++d)
    {
        for (size_t i=0; i<inputSize; ++i)
        {
            float val=0.;
            for (size_t o=0; o<outputSize; ++o)
                val += delta[d*outputSize+o]*tmp[i][o];
            
            prevDelta[d*inputSize+i] = prevAFunc.df(prevA[d*inputSize+i])*val*prevDrop[d*inputSize+i];
        }
    }
}
//
void FCLayer::calcGrad()
{
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
