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
    // A_(l+1)  = AFunc( W_(l+1) A_l + B_(l+1))
    // dA_(l+1) = dAFunc(A_(l+1))
    const std::vector<double>& prevA = prevLayer->getA();
    
    for (size_t d=0; d<nbData; ++d)
    {
        for (size_t o=0; o<outputSize; ++o)
        {
            double val=bias[o];
            for (size_t i=0; i<inputSize; ++i)
                val+= weight[o*inputSize+i]*prevA[d*inputSize+i];
            
            val   = AFunc.f(val);
            
            a[d*outputSize+o]  = val;
            da[d*outputSize+o] = AFunc.df(val);
        }
    }
}
//
void FCLayer::bwdProp()
{
    calcGrad();
    
    // D_l = (W'_(l+1) D(l+1)) . dA_l
    const std::vector<double>& prevdA = prevLayer->getdA();
    std::vector<double>& prevDelta    = prevLayer->getDelta();
    
    for (size_t d=0; d<nbData; ++d)
    {
        for (size_t i=0; i<inputSize; ++i)
        {
            double val=0.;
            for (size_t o=0; o<outputSize; ++o)
                val += delta[d*outputSize+o]*weight[o*inputSize+i];
            
            prevDelta[d*inputSize+i] = prevdA[d*inputSize+i]*val;
        }
    }
}
//
void FCLayer::calcGrad()
{
    const std::vector<double>& prevA = prevLayer->getA();
    
    for (size_t d=0; d<nbData; ++d)
    {
        for (size_t o=0; o<outputSize; ++o)
        {
            dbias[o] += delta[d*outputSize+o];
            for (size_t i=0; i<inputSize; ++i)
                dweight[o*inputSize+i] += delta[d*outputSize+o] * prevA[d*inputSize+i];
        }
    }
}
//
