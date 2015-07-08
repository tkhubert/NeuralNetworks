//
//  FCLayer.cpp
//  NeuralNetworks
//
//  Created by Thomas Hubert on 04/07/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#include "FCLayer.h"

//
void FCLayer::fwdProp()
{
    const std::vector<double>& prevA = prevLayer->getA();
    
    for (size_t o=0; o<outputSize; ++o)
    {
        double val=bias[o];
        for (size_t i=0; i<inputSize; ++i)
            val+= weight[o*inputSize+i]*prevA[i];
        
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
//
