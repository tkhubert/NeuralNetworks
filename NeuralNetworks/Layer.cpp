//
//  Layer.cpp
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#include "Layer.h"
size_t Layer::layerCount = 0;
//
Layer::Layer(size_t _inputSize, size_t _outputSize, const ActivationFunc& _AFunc) :
    inputSize(_inputSize),
    outputSize(_outputSize),
    AFunc(_AFunc)
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
    a.resize      (outputSize*nbData);
    delta.resize  (outputSize*nbData);
}
//
void Layer::setDCost(const std::vector<float> &dc)
{
    for (size_t i=0; i<outputSize*nbData; ++i)
        delta[i] = AFunc.df(a[i])*dc[i];
}
//
void Layer::initParams()
{
    if (inputSize==0)
        return;
    
    std::default_random_engine      gen((int)layerNb);
    std::normal_distribution<float> norm(0.,1.);
    
    for (size_t i=0; i<bias.size(); ++i)
        bias[i]   = norm(gen);
    
    float normalizer = 1./sqrt(inputSize);
    for (size_t i=0; i<weight.size(); ++i)
        weight[i] = norm(gen)*normalizer;
}
//
void Layer::updateParams(float alpha, float friction, float lambda)
{
    for (size_t i=0; i<bias.size(); ++i)
    {
        float vtmp = vbias[i];
        vbias[i]  = friction*vbias[i] - alpha*dbias[i];
        bias[i]  += -friction*vtmp + (1+friction) * vbias[i];
        dbias[i]  = 0.;
    }
    for (size_t i=0; i<weight.size(); ++i)
    {
        float vtmp = vweight[i];
        vweight[i]  = friction*vweight[i] - alpha*(dweight[i]+lambda*weight[i]);
        weight[i]  += -friction*vtmp + (1+friction)*vweight[i];
        dweight[i]  = 0.;
    }
}
