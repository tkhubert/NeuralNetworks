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
Layer::LayerParams::LayerParams(size_t nbBias, size_t nbWeight)
{
    params.resize(nbBias+nbWeight);
    bias   = innerData(params.begin(), params.begin()+nbBias);
    weight = innerData(params.begin()+nbBias, params.end());
}
//
void Layer::LayerParams::resize(size_t nbBias, size_t nbWeight)
{
    params.resize(nbBias+nbWeight);
    bias   = innerData(params.begin(), params.begin()+nbBias);
    weight = innerData(params.begin()+nbBias, params.end());
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
    
    gen.seed((int) layerNb);
}
//
Layer::~Layer()
{
    --layerCount;
    
    prevLayer = nullptr;
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
void Layer::setDCost(vec_r&& dc)
{
    for (size_t i=0; i<delta.size(); ++i)
        delta[i] = AFunc.df(a[i])*dc[i];
}
//
void Layer::regularize(real lambda)
{
    transform(params.weight.begin(), params.weight.end(), dparams.weight.begin(), dparams.weight.begin(), [l=lambda] (auto w, auto dw) {return dw+l*w;});
}
//
void Layer::initParams()
{
    normal_distribution<real> norm(0.,1.);
    
    for (auto bItr=params.bias.begin(); bItr!=params.bias.end(); ++bItr)
        *bItr = norm(gen);
    
    real normalizer = 1./sqrt(weightInputSize);
    for (auto wItr=params.weight.begin(); wItr!=params.weight.end(); ++wItr)
        *wItr = norm(gen)*normalizer;
}
//
void Layer::updateParams(Optimizer& optim)
{
    optim.updateParams(params.params, dparams.params);
}
//
}