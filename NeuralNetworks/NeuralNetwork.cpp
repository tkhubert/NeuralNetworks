//
//  NeuralNetwork.cpp
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#include "NeuralNetwork.h"

template<typename ActFunc, typename CtFunc>
void NeuralNetwork<ActFunc, CtFunc>::addLayer(Layer<ActFunc>& layer)
{
    if (nbLayers>0)
    {
        Layer<ActFunc>& pLayer = layers[nbLayers-1];
        pLayer.setNextLayer(layer);
        layer.setPrevLayer (pLayer);
    }
    layers.push_back(layer);
    nbLayers++;
}