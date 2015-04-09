//
//  Layer.cpp
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#include "Layer.h"

Layer::Layer(size_t _size, const ActivationFunc& _AFunc) : size(_size), AFunc(_AFunc)
{
    a.resize(size);
    da.resize(size);
    delta.resize(size);
    
    bias.resize(size);
    
    prevLayer = nullptr;
    nextLayer = nullptr;
}
Layer::~Layer()
{
    prevLayer = nullptr;
    nextLayer = nullptr;
}
//
//void FCLayer::setDCost(const std::vector<double> &dc)
//{
//    for (size_t i=0; i<size; ++i)
//        delta[i] = da[i]*dc[i];
//}
//
void FCLayer::fwdProp()
{
    size_t iSize = prevLayer->getSize();
    size_t oSize = size;
    
    const std::vector<double>& inputA = prevLayer->getA();
    
    for (size_t i=0; i<oSize; ++i)
    {
        double val=0.;
        for (size_t j=0; j<iSize; ++j)
            val+= weight[i*iSize+j]*inputA[j];
        
        val  += bias[i];
        val   = AFunc.f(val);
        
        a[i]  = val;
        da[i] = AFunc.df(val);
    }
}
//
void FCLayer::bwdProp()
{
    size_t iSize = size;
    size_t oSize = prevLayer->getSize();
    
    const std::vector<double>& outputdA    = prevLayer->getdA();
    std::vector<double> outputDelta(oSize);
    
    for (size_t i=0; i<oSize; ++i)
    {
        double val=0.;
        for (size_t j=0; j<iSize; ++j)
            val += weight[i*iSize+j]*delta[j];
        
        val *= outputdA[i];
        
        outputDelta[i] = val;
    }
    
    prevLayer->setDelta(outputDelta);
}