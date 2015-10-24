//
//  Layer.cpp
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#include "Layer.h"
namespace NN {
    
size_t Layer::layerCount = 0;
//
Layer::Layer(size_t size, real dropRate, const ActivationFunc& AFunc) :
    inputSize(0),
    outputSize(size),
    dropRate(dropRate),
    phase(Phase::TEST),
    AFunc(AFunc)
{
    layerNb = layerCount++;
    
    prevLayer = nullptr;
    nextLayer = nullptr;
    
    gen.seed((int) layerNb);
}
//
Layer::~Layer()
{
    --layerCount;
    
    prevLayer = nullptr;
    nextLayer = nullptr;
}
//
void Layer::resize(size_t _nbData)
{
    nbData = _nbData;
    a.resize    (outputSize*nbData);
    delta.resize(outputSize*nbData);
    drop.resize (outputSize*nbData);
}
//
void Layer::setDrop()
{
    if (phase==Phase::TEST)
    {
        fill(drop.begin(), drop.end(), 1.-dropRate);
    }
    else
    {
        bernoulli_distribution bern(1.-dropRate);
        
        for (size_t i=0; i<drop.size(); ++i)
            drop[i] = bern(gen);
    }
}
//
void Layer::setDCost(const vec_r &dc)
{
    for (size_t i=0; i<delta.size(); ++i)
        delta[i] = AFunc.df(a[i])*dc[i];
}
//
void Layer::regularize(real lambda)
{
    auto nbWeight = params.getNbWeight();
    auto dweight  = dparams.getWPtr();
    const auto weight = params.getCWPtr();
    
    transform(weight, weight+nbWeight, dweight, dweight, [l=lambda] (auto w, auto dw) {return dw+l*w;});
}
//
}