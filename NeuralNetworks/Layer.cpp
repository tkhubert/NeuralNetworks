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
Layer::Layer(size_t inputSize, size_t outputSize, float dropRate, const ActivationFunc& AFunc) :
    inputSize(inputSize),
    outputSize(outputSize),
    dropRate(dropRate),
    phase(Phase::TEST),
    AFunc(AFunc)
{
    layerNb = layerCount++;
    
    bias.resize   (outputSize);
    dbias.resize  (outputSize);
    vbias.resize  (outputSize);
    weight.resize (inputSize*outputSize);
    dweight.resize(inputSize*outputSize);
    vweight.resize(inputSize*outputSize);
    
    prevLayer = nullptr;
    nextLayer = nullptr;
    
    gen.seed((int) layerNb);
    initParams();
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
//        if (layerNb==0 || layerNb==layerCount-1)
//        {
//            fill(drop.begin(), drop.end(), 1.);
//            return;
//        }
//        
//        for (size_t d=0; d<nbData; ++d)
//        {
//            for (size_t i=0; i<50; ++i)
//                drop[d*100+i] = 1;
//            for (size_t i=50; i<100; ++i)
//                drop[d*100+i] = 0;
//        }
        
        bernoulli_distribution bern(1.-dropRate);
        
        for (size_t i=0; i<drop.size(); ++i)
            drop[i] = bern(gen);
    }
}
//
void Layer::setDCost(const vector<float> &dc)
{
    for (size_t i=0; i<outputSize*nbData; ++i)
        delta[i] = AFunc.df(a[i])*dc[i];
}
//
void Layer::initParams()
{
    if (inputSize==0)
        return;
    
    normal_distribution<float> norm(0.,1.);
    
    for (size_t o=0; o<outputSize; ++o)
        bias[o] = norm(gen);
    
    float normalizer = 1./sqrt(inputSize);
    for (size_t o=0; o<outputSize; ++o)
        for (size_t i=0; i<inputSize; ++i)
            weight[o*inputSize+i] = norm(gen)*normalizer;
}
//
void Layer::updateParams(float alpha, float friction, float lambda)
{
    for (size_t o=0; o<outputSize; ++o)
    {
        auto vtmp = vbias[o];
        vbias[o]  = friction*vbias[o] - alpha*dbias[o];
        bias[o]  += -friction*vtmp + (1+friction) * vbias[o];
        dbias[o]  = 0.;
    }
    
    for (size_t o=0; o<outputSize; ++o)
    {
        for (size_t i=0; i<inputSize; ++i)
        {
            auto idx      = o*inputSize+i;
            auto vtmp     = vweight[idx];
            vweight[idx]  = friction*vweight[idx] - alpha*(dweight[idx]+lambda*weight[idx]);
            weight[idx]  += -friction*vtmp + (1+friction)*vweight[idx];
            dweight[idx]  = 0.;
        }
    }
}

}