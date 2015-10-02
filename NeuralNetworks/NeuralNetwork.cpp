//
//  NeuralNetwork.cpp
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#include "NeuralNetwork.h"

namespace NN {
    
NeuralNetwork::NeuralNetwork(const CostFunc& CFunc, vector<unique_ptr<Layer>>&& _layers) :
    CFunc(CFunc),
    nbLayers(_layers.size()),
    layers(move(_layers))
{
    for (size_t i=1; i<nbLayers; ++i)
    {
        auto& pLayer = layers[i-1];
        auto& cLayer = layers[i];
        
        pLayer->setNextLayer(cLayer.get());
        cLayer->setPrevLayer(pLayer.get());
    }
    
    inputSize  = layers.front()->getOutputSize();
    outputSize = layers.back()->getOutputSize();
    
}
//
string NeuralNetwork::getName() const
{
    stringstream ss;
    for (size_t i=0; i<nbLayers; ++i)
        ss << layers[i]->getOutputSize() << "_";

    ss << CFunc.getName() << "_" << layers.back()->getAFunc().getName();
    return ss.str();
}
//
void NeuralNetwork::setNbData(size_t nbData)
{
    for (size_t i=0; i<nbLayers; ++i)
        layers[i]->setNbData(nbData);
}
//
void NeuralNetwork::setPhase(Phase phase)
{
    for (size_t i=0; i<nbLayers; ++i)
        layers[i]->setPhase(phase);
}
//
void NeuralNetwork::setDrop()
{
    for (size_t i=0; i<nbLayers; ++i)
        layers[i]->setDrop();
}
//
void NeuralNetwork::updateParams(Optimizer& optim)
{
    for (size_t i=1; i<nbLayers; ++i)
        optim.updateParams(*layers[i]);
}
//
void NeuralNetwork::setInput(const LabelData& lD)
{
    setNbData(1);
    layers.front()->setA(lD.data);
}
//
void NeuralNetwork::setInput(LabelDataCItr dataStart, LabelDataCItr dataEnd)
{
    auto nbData   = distance(dataStart, dataEnd);
    auto dataSize = dataStart->data.size();
    
    vector<float> input(nbData*dataSize);
    for (size_t d=0; d<nbData; ++d)
    {
        auto data = (dataStart+d)->data;
        copy(data.begin(), data.end(), input.begin()+d*inputSize);
    }
    
    setNbData(nbData);
    layers.front()->setA(move(input));
}
//
void NeuralNetwork::fwdProp(const LabelData& lD)
{
    setInput(lD);
    setDrop();
    for (size_t i=0; i<nbLayers; ++i)
        layers[i]->fwdProp();
}
//
void NeuralNetwork::fwdProp(LabelDataCItr dataStart, LabelDataCItr dataEnd)
{
    setInput(dataStart, dataEnd);
    setDrop();
    for (size_t i=0; i<nbLayers; ++i)
        layers[i]->fwdProp();
}
//
void NeuralNetwork::bwdProp(const vector<float>& dC)
{
    setDCost(dC);
    for (size_t i=nbLayers-1; i>=2; --i)
        layers[i]->bwdProp();
}
//
void NeuralNetwork::calcGrad()
{
    for (size_t i=nbLayers-1; i>=1; --i)
        layers[i]->calcGrad();
}
//
float NeuralNetwork::calcCost(LabelDataCItr dataStart, LabelDataCItr dataEnd) const
{
    return CFunc.f(getOutput(), dataStart, dataEnd);
}
//
void NeuralNetwork::calcDCost(LabelDataCItr dataStart, LabelDataCItr dataEnd, vector<float>& dC)
{
    return CFunc.df(getOutput(), dataStart, dataEnd, dC);
}
//
const auto& NeuralNetwork::predict(const LabelData& lD)
{
    fwdProp(lD);
    return getOutput();
}
//
size_t NeuralNetwork::isCorrect(LabelDataCItr dataStart, LabelDataCItr dataEnd) const
{
    const auto& prediction = getOutput();
    auto nbData = distance(dataStart, dataEnd);
    
    size_t nbCorrect = 0;
    for (size_t d=0; d<nbData; ++d)
    {
        auto s = prediction.cbegin()+d*outputSize;
        auto e = s + outputSize;
        const LabelData& lD = *(dataStart+d);
        
        nbCorrect += distance(s, max_element(s, e))==lD.label;
    }
    return nbCorrect;
}
//
void NeuralNetwork::train(const DataContainer& data, Optimizer& optim)
{
    auto lData = data.getTrainLabelData();
    
    auto name = getName() + "_" + optim.getName();
    ofstream debugFile  = ofstream("/Users/tkhubert/Documents/Projects/NeuralNetworks/MNist/"+name+".csv");

    cout << "Start training " << name << "-------------"<<endl;
    
    auto nbEpochs   = optim.getNbEpochs();
    auto batchSize  = optim.getBatchSize();
    auto totalISize = lData.size();
    auto nbBatches  = (totalISize-1)/batchSize + 1;
    
    optim.resize(layers);
    
    for (size_t t=0; t<nbEpochs; ++t)
    {
        debugFile << t << ", ";
        cout << "Epoch: " << t << ", ";
        clock_t startTimeEpoch = clock();
        
        setPhase(Phase::TRAIN);
        random_shuffle(lData.begin(), lData.end());
        
        for (size_t batch=0; batch<nbBatches; ++batch)
        {
            auto start  = batch*batchSize;
            auto end    = min(start+batchSize, lData.size());
            auto nbData = end-start;
            
            vector<float> dC(outputSize*nbData);
            auto dataStart = lData.cbegin()+start;
            auto dataEnd   = dataStart+nbData;
            
            fwdProp  (dataStart, dataEnd);
            calcDCost(dataStart, dataEnd, dC);
            bwdProp  (dC);
            calcGrad ();
            
            updateParams(optim);
        }
        
        auto timeEpoch = ( clock() - startTimeEpoch ) / (float) CLOCKS_PER_SEC;
        debugFile << "time " << timeEpoch << "s,";
        cout << "time " << timeEpoch << "s,";
        
        test(data.getTrainLabelData(), batchSize);
        auto trainErrRate = errRate;
        auto trainCost    = cost;
        
        test(data.getCrossLabelData(), batchSize);
        auto crossErrRate = errRate;
        auto crossCost    = cost;
        
        test(data.getTestLabelData(), batchSize);
        auto testErrRate = errRate;
        auto testCost    = cost;
        
        debugFile << trainErrRate << "," << crossErrRate << "," << testErrRate << ",";
        debugFile << trainCost    << "," << crossCost    << "," << testCost    << endl;
        cout      << trainErrRate << "," << crossErrRate << "," << testErrRate << ",";
        cout      << trainCost    << "," << crossCost    << "," << testCost    << endl;
    }
    
    debugFile.close();
}
//
void NeuralNetwork::test(const vector<LabelData>& lData, size_t batchSize)
{
    setPhase(Phase::TEST);
    
    cost   =0.;
    errRate=0.;
    
    auto totalISize = lData.size();
    auto nbBatches  = (totalISize-1)/batchSize + 1;
    
    for (size_t batch=0; batch<nbBatches; ++batch)
    {
        auto start  = batch*batchSize;
        auto end    = min(start+batchSize, lData.size());
        auto nbData = end-start;
        
        auto dataStart = lData.cbegin()+start;
        auto dataEnd   = dataStart+nbData;
        
        fwdProp(dataStart, dataEnd);
        cost    += calcCost (dataStart, dataEnd);
        errRate += isCorrect(dataStart, dataEnd);
    }
    
    cost    /= totalISize;
    errRate  = 1.-errRate/totalISize;
}

}