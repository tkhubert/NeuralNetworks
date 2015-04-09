//
//  Layer.cpp
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#include "Layer.h"

Layer::Layer(size_t _inputSize, size_t _outputSize, const ActivationFunc& _AFunc) : inputSize(_inputSize), outputSize(_outputSize), AFunc(_AFunc)
{
    a.resize      (outputSize);
    bias.resize   (outputSize);
    weight.resize (inputSize*outputSize);
    
    da.resize     (outputSize);
    dbias.resize  (outputSize);
    dweight.resize(inputSize*outputSize);
    delta.resize  (outputSize);
    
    prevLayer = nullptr;
    nextLayer = nullptr;
}
Layer::~Layer()
{
    prevLayer = nullptr;
    nextLayer = nullptr;
}
//
void Layer::setDCost(const std::vector<double> &dc)
{
    for (size_t i=0; i<outputSize; ++i)
        delta[i] = da[i]*dc[i];
}
//
void FCLayer::fwdProp()
{
    const std::vector<double>& prevA = prevLayer->getA();
    
    for (size_t o=0; o<outputSize; ++o)
    {
        double val=0.;
        for (size_t i=0; i<inputSize; ++i)
            val+= weight[o*inputSize+i]*prevA[i];
        
        val  += bias[o];
        val   = AFunc.f(val);
        
        a[o]  = val;
        da[o] = AFunc.df(val);
    }
}
//
void FCLayer::bwdProp()
{
    const std::vector<double>& prevdA = prevLayer->getdA();
    std::vector<double> prevDelta(inputSize);
    
    for (size_t i=0; i<inputSize; ++i)
    {
        double val=0.;
        for (size_t o=0; o<outputSize; ++o)
            val += weight[i*inputSize+o]*delta[o];
        
        prevDelta[i] = prevdA[i]*val;
    }
    
    prevLayer->setDelta(prevDelta);
}