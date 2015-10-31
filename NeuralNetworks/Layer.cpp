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
    for_each(drop.begin(), drop.end(), [&b=bern, &g=gen] (auto& d) {d=b(g);});
}
//
void Layer::regularize(real lambda)
{
    transform(params.weight.begin(), params.weight.end(), dparams.weight.begin(), dparams.weight.begin(), [lambda] (auto w, auto dw) {return dw+lambda*w;});
}
//
void Layer::initParams()
{
    normal_distribution<real> norm(0.,1.);
    for_each(params.begin(), params.end(), [&n=norm, &g=gen] (auto& p) {p=n(g);});
    for_each(params.weight.begin(), params.weight.end(), [sig=1./sqrt(weightInputSize)] (auto& w) {w*=sig;});
}
//
void Layer::updateParams(Optimizer& optim)
{
    optim.updateParams(params.params, dparams.params);
}
//
}