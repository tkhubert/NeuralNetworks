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
Layer::Params::Params(size_t nbBias, size_t nbWeight, size_t weightInputSize):
    nbData(nbBias+nbWeight),
    nbBias(nbBias),
    nbWeight(nbWeight),
    weightInputSize(weightInputSize)
{
    params.resize(nbData);
}
//
void Layer::Params::resize(size_t _nbBias, size_t _nbWeight, size_t _weightInputSize)
{
    nbData          = _nbBias+_nbWeight;
    nbBias          = _nbBias;
    nbWeight        = _nbWeight;
    weightInputSize = _weightInputSize;
    params.resize(nbData);
}
//
    
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
    if (dropRate<TINY || phase==Phase::TEST)
    {
        fill(drop.begin(), drop.end(), 1.-dropRate);
        return;
    }
    
    bernoulli_distribution bern(1.-dropRate);
    for (size_t i=0; i<drop.size(); ++i)
        drop[i] = bern(gen);
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
    auto nbWeight = params.nbWeight;
    auto dweight  = dparams.getWPtr();
    const auto weight = params.getCWPtr();
    
    transform(weight, weight+nbWeight, dweight, dweight, [l=lambda] (auto w, auto dw) {return dw+l*w;});
}
//
void Layer::initParams()
{
    normal_distribution<real> norm(0.,1.);
    real normalizer = 1./sqrt(params.weightInputSize);
    
    size_t o=0;
    for (; o<params.nbBias; ++o)
        params.params[o] = norm(gen);
    for (; o<params.nbData; ++o)
        params.params[o] = norm(gen)*normalizer;
}
//
}