//
//  NeuralNetwork.cpp
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const CostFunc& _CFunc, const Optimizer& _Optim, std::vector<Layer>& _layers) : nbLayers(0) , CFunc(_CFunc), Optim(_Optim)
{
    nbLayers = _layers.size();
    layers.resize(nbLayers);
    
    layers.push_back(&_layers[0]);
    for (size_t i=1; i<nbLayers; ++i)
    {
        Layer* pLayer = layers[i-1];
        Layer* cLayer = &_layers[i];
        assert(pLayer->getOutputSize()==cLayer->getInputSize());
        
        pLayer->setNextLayer(cLayer);
        cLayer->setPrevLayer(pLayer);
        layers[i] = cLayer;
    }
    
    inputSize  = layers[0]->getInputSize();
    outputSize = layers[nbLayers-1]->getInputSize();
}
//
void NeuralNetwork::fwdProp()
{
    for (size_t i=0; i<nbLayers; ++i)
        layers[i]->fwdProp();
}
//
void NeuralNetwork::bwdProp()
{
    for (size_t i=nbLayers-1; i>=1; ++i)
        layers[i]->bwdProp();
}
//
const std::vector<double>& NeuralNetwork::predict(const std::vector<double>& inputs)
{
    setInput(inputs);
    fwdProp();
    return getOuptut();
}
//
void NeuralNetwork::train(const std::vector<std::vector<double> >& inputs, const std::vector<std::vector<double> >& labels)
{
    size_t nbBatches  = (inputs.size()-1)/Optim.batchSize + 1;
    
    for (size_t batch=0; batch<nbBatches; ++batch)
    {
        size_t start = batch*Optim.batchSize;
        size_t end   = std::min(start+Optim.batchSize, inputs.size());
        
        std::vector<double> dc(outputSize);
        for (size_t i=start; i<end; ++i)
        {
            setInput(inputs[i]);
            fwdProp();
            
            for (size_t j=0; j<outputSize; ++j)
                dc[j] += calcDCost(j, labels[i]);
        }
        
        for (size_t j=0; j<outputSize; ++j)
            dc[j] /= (end-start);
        
        setDCost(dc);
        bwdProp();
    }
}
//
void NeuralNetwork::test(const std::vector<std::vector<double> >& inputs, const std::vector<std::vector<double> >& labels)
{
    double cost=0.;
    for (size_t i=0; i<inputs.size(); ++i)
    {
        predict(inputs[i]);
        cost += calcCost(labels[i]);
    }
    
    c = cost/inputs.size();
}