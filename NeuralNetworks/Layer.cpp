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
    
    initParams();
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
void Layer::initParams()
{
    std::default_random_engine       gen;
    std::normal_distribution<double> norm(0.,1.0);
    
    for (size_t i=0; i<bias.size(); ++i)
        bias[i]   = norm(gen);
    for (size_t i=0; i<weight.size(); ++i)
        weight[i] = norm(gen);
}
void Layer::updateParams(double alpha)
{
    for (size_t i=0; i<bias.size(); ++i)
    {
        bias[i]  -= alpha*dbias[i];
        dbias[i]  = 0.;
    }
    for (size_t i=0; i<weight.size(); ++i)
    {
        weight[i] -= alpha*dweight[i];
        dweight[i] = 0.;
    }
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
    calcGrad();
    
    const std::vector<double>& prevdA = prevLayer->getdA();
    std::vector<double>& prevDelta    = prevLayer->getDelta();
    
    for (size_t i=0; i<inputSize; ++i)
    {
        double val=0.;
        for (size_t o=0; o<outputSize; ++o)
            val += delta[o]*weight[o*inputSize+i];
        
        prevDelta[i] = prevdA[i]*val;
    }
}
//
void FCLayer::calcGrad()
{
    const std::vector<double>& prevA = prevLayer->getA();
    
    for (size_t o=0; o<outputSize; ++o)
    {
        dbias[o] += delta[o];
        for (size_t i=0; i<inputSize; ++i)
            dweight[o*inputSize+i] += delta[o] * prevA[i];
    }
}