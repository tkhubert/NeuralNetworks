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
    const std::vector<float>& prevA = prevLayer->getA();
    
    for (size_t d=0; d<nbData; ++d)
    {
        for (size_t o=0; o<outputSize; ++o)
        {
            float val=bias[o];
            for (size_t i=0; i<inputSize; ++i)
                val+= weight[o*inputSize+i]*prevA[d*inputSize+i];
            
            val = AFunc.f(val);
            
            a[d*outputSize+o]  = val;
        }
    }
}
//
void FCLayer::bwdProp()
{
    calcGrad();
    
    // D_l = (W'_(l+1) D(l+1)) . dA_l
    const std::vector<float>& prevA = prevLayer->getA();
    std::vector<float>& prevDelta   = prevLayer->getDelta();
    
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
            
            prevDelta[d*inputSize+i] = AFunc.df(prevA[d*inputSize+i])*val;
        }
    }
}
//
void FCLayer::calcGrad()
{
    const std::vector<float>& prevA = prevLayer->getA();
    
    for (size_t d=0; d<nbData; ++d)
    {
        for (size_t o=0; o<outputSize; ++o)
        {
            float tmpDelta = delta[d*outputSize+o];
            
            dbias[o] += tmpDelta;
            for (size_t i=0; i<inputSize; ++i)
                dweight[o*inputSize+i] += tmpDelta * prevA[d*inputSize+i];
        }
    }
}
//
