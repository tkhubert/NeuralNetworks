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
void NeuralNetwork::initWeights()
{
    for (size_t i=0; i<nbLayers; ++i)
        layers[i]->initWeights();
}
//
void NeuralNetwork::updateWeights()
{
    for (size_t i=0; i<nbLayers; ++i)
        layers[i]->updateWeights(Optim.alpha);
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
    return getOutput();
}
//
bool NeuralNetwork::isCorrect(int label) const
{
    const std::vector<double>& prediction = getOutput();
    return std::distance(prediction.begin(), std::max_element(prediction.begin(), prediction.end()))==label;
}
//
void NeuralNetwork::train(const DataContainer& data)
{
    const std::vector<std::vector<double> >& inputs = data.getTrainData();
    const std::vector<int>&                  labels = data.getTrainLabels();
    
    std::cout << "Start training -------------------------------------"<<std::endl;
    
    size_t nbBatches  = (inputs.size()-1)/Optim.batchSize + 1;
    
    for (size_t t=0; t<Optim.nbEpochs; ++t)
    {
        std::cout << "   Epoch: " << t << "-------------------------------" << std::endl;
        std::clock_t startTimeEpoch = std::clock();;
        
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
            updateWeights();
        }
        
        double timeEpoch = ( std::clock() - startTimeEpoch ) / (double) CLOCKS_PER_SEC;
        std::cout << "     time to train for epoch: " << timeEpoch << std::endl;
        
        test(inputs, labels);
        std::cout << "     Train: errRate is" << errRate << ", cost is" << cost << std::endl;
        test(data.getCrossData(), data.getCrossLabels());
        std::cout << "     Cross: errRate is" << errRate << ", cost is" << cost << std::endl;
    }
}
//
void NeuralNetwork::test(const std::vector<std::vector<double> >& inputs, const std::vector<int>& labels)
{
    cost=0.;
    errRate=0.;
    
    for (size_t i=0; i<inputs.size(); ++i)
    {
        predict(inputs[i]);
        cost += calcCost(labels[i]);
        errRate += isCorrect(labels[i]);
    }
    
    cost    /= inputs.size();
    errRate  = (1.-errRate)/inputs.size();
}