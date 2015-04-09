//
//  NeuralNetwork.cpp
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#include "NeuralNetwork.h"

void NeuralNetwork::addLayer(Layer& layer)
{
    if (nbLayers>0)
    {
        Layer& pLayer = layers[nbLayers-1];
        pLayer.setNextLayer(&layer);
        layer.setPrevLayer (&pLayer);
    }
    layers.push_back(layer);
    nbLayers++;
}
//
void NeuralNetwork::fwdProp()
{
    for (size_t i=0; i<nbLayers; ++i)
        layers[i].fwdProp();
}
//
void NeuralNetwork::bwdProp()
{
    for (size_t i=nbLayers-1; i>=1; ++i)
        layers[i].bwdProp();
}
//
const std::vector<double>& NeuralNetwork::predict(const std::vector<double>& inputs)
{
    layers[0].setA(inputs);
    fwdProp();
    return layers[nbLayers-1].getA();
}
//
void NeuralNetwork::train(const std::vector<std::vector<double> >& inputs, const std::vector<double>& labels)
{
    
}
//
void NeuralNetwork::test(const std::vector<std::vector<double> >& inputs, const std::vector<double>& labels) const
{
    
}