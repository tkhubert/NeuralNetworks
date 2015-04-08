//
//  Layer.cpp
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#include "Layer.h"

template<typename ActFunc>
Layer<ActFunc>::Layer(size_t _size) : size(_size)
{
    a.resize(size);
    da.resize(size);
    delta.resize(size);
    
    bias.resize(size);
    
    prevLayer = nullptr;
    nextLayer = nullptr;
}
//
template<typename ActFunc>
void FCLayer<ActFunc>::fwdProp()
{
    const std::vector<double>& inputA = this->prevLayer->a;
    
    size_t iSize = this->prevLayer->size;
    size_t oSize = this->size;
    
    for (size_t i=0; i<oSize; ++i)
    {
        double val=0.;
        for (size_t j=0; j<iSize; ++j)
            val+= this->weight[i*iSize+j]*inputA[j];
        
        val  += this->bias[i];
        val   = this->AFunc.f(val);
        
        this->a[i]  = val;
        this->da[i] = this->AFunc.df(val);
    }
}
//
template<typename ActFunc>
void FCLayer<ActFunc>::bwdProp()
{
    const std::vector<double>& outputdA    = this->prevLayer->da;
    std::vector<double>&       outputDelta = this->prevLayer->delta;
    
    size_t iSize = this->size;
    size_t oSize = this->prevLayer->size;
    
    for (size_t i=0; i<oSize; ++i)
    {
        double val=0.;
        for (size_t j=0; j<iSize; ++j)
            val += this->weight[i*iSize+j]*this->delta[j];
        
        val *= outputdA[i];
        
        outputDelta[i] = val;
    }
}