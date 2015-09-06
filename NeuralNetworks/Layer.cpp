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
Layer::Layer(size_t size, float dropRate, const ActivationFunc& AFunc) :
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
void Layer::setDCost(const vector<float> &dc)
{
    for (size_t i=0; i<delta.size(); ++i)
        delta[i] = AFunc.df(a[i])*dc[i];
}
//
void Layer::initParams()
{
    if (inputSize==0)
        return;
    
    normal_distribution<float> norm(0.,1.);
    
    for (size_t o=0; o<bias.size(); ++o)
        bias[o] = norm(gen);
    
    float normalizer = 1./sqrt(inputSize);
    for (size_t o=0; o<weight.size(); ++o)
        weight[o] = norm(gen)*normalizer;
}
//
void Layer::updateParams(float alpha, float friction, float lambda)
{
    for (size_t o=0; o<bias.size(); ++o)
    {
        auto vtmp = vbias[o];
        vbias[o]  = friction*vbias[o] - alpha*dbias[o];
        bias [o] += -friction*vtmp + (1+friction)*vbias[o];
        dbias[o]  = 0.;
    }
    
    for (size_t o=0; o<weight.size(); ++o)
    {
        auto vtmp   = vweight[o];
        vweight[o]  = friction*vweight[o] - alpha*(dweight[o]+lambda*weight[o]);
        weight [o] += -friction*vtmp + (1+friction)*vweight[o];
        dweight[o]  = 0.;
    }
}

}