//
//  NeuralNetwork.cpp
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#include "NeuralNetwork.h"

namespace NN {
    
NeuralNetwork::NeuralNetwork(const CostFunc& CFunc, const Optimizer& Optim, vector<unique_ptr<Layer>>&& _layers) :
    CFunc(CFunc),
    Optim(Optim),
    nbLayers(_layers.size()),
    layers(move(_layers))
{
    for (size_t i=1; i<nbLayers; ++i)
    {
        auto& pLayer = layers[i-1];
        auto& cLayer = layers[i];
        assert(pLayer->getOutputSize()==cLayer->getInputSize());
        
        pLayer->setNextLayer(cLayer.get());
        cLayer->setPrevLayer(pLayer.get());
    }
    
    inputSize  = layers.front()->getOutputSize();
    outputSize = layers.back()->getOutputSize();
    
    debugFile  = ofstream("/Users/tkhubert/Documents/Projects/NeuralNetworks/MNist/"+getName());
}
//
string NeuralNetwork::getName() const
{
    stringstream ss;
    for (size_t i=0; i<nbLayers; ++i)
        ss << layers[i]->getOutputSize() << "_";

    ss << CFunc.getName() << "_" << layers.back()->getAFunc().getName() << "_" << Optim.getName();
    ss << ".csv";

    return ss.str();
}
//
void NeuralNetwork::setNbData(size_t nbData)
{
    for (size_t i=0; i<nbLayers; ++i)
        layers[i]->setNbData(nbData);
}
//
void NeuralNetwork::updateParams()
{
    for (size_t i=1; i<nbLayers; ++i)
        layers[i]->updateParams(Optim.alpha, Optim.friction, Optim.lambda);
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
    for (size_t i=1; i<nbLayers; ++i)
        layers[i]->fwdProp();
}
//
void NeuralNetwork::fwdProp(LabelDataCItr dataStart, LabelDataCItr dataEnd)
{
    setInput(dataStart, dataEnd);
    for (size_t i=1; i<nbLayers; ++i)
        layers[i]->fwdProp();
}
//
void NeuralNetwork::bwdProp(const vector<float>& dC)
{
    setDCost(dC);
    for (size_t i=nbLayers-1; i>=1; --i)
        layers[i]->bwdProp();
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
void NeuralNetwork::train(const DataContainer& data)
{
    auto lData = data.getTrainLabelData();
    
    cout << "Start training " << getName() << "-------------"<<endl;
    
    auto totalISize = lData.size();
    auto nbBatches  = (totalISize-1)/Optim.batchSize + 1;
    
    for (size_t t=0; t<Optim.nbEpochs; ++t)
    {
        debugFile << t << ", ";
        cout << "Epoch: " << t << ", ";
        clock_t startTimeEpoch = clock();
        
        random_shuffle(lData.begin(), lData.end());
        
        for (size_t batch=0; batch<nbBatches; ++batch)
        {
            auto start  = batch*Optim.batchSize;
            auto end    = min(start+Optim.batchSize, lData.size());
            auto nbData = end-start;
            
            vector<float> dC(outputSize*nbData);
            auto dataStart = lData.cbegin()+start;
            auto dataEnd   = dataStart+nbData;
            
            fwdProp  (dataStart, dataEnd);
            calcDCost(dataStart, dataEnd, dC);
            bwdProp  (dC);
            
            updateParams();
        }
        
        auto timeEpoch = ( clock() - startTimeEpoch ) / (float) CLOCKS_PER_SEC;
        debugFile << "time " << timeEpoch << "s,";
        cout << "time " << timeEpoch << "s,";
        
        test(data.getTrainLabelData());
        auto trainErrRate = errRate;
        auto trainCost    = cost;
        
        test(data.getCrossLabelData());
        auto crossErrRate = errRate;
        auto crossCost    = cost;
        
        test(data.getTestLabelData());
        auto testErrRate = errRate;
        auto testCost    = cost;
        
        debugFile << trainErrRate << "," << crossErrRate << "," << testErrRate << ",";
        debugFile << trainCost    << "," << crossCost    << "," << testCost    << endl;
        cout << trainErrRate << "," << crossErrRate << "," << testErrRate << ",";
        cout << trainCost    << "," << crossCost    << "," << testCost    << endl;
    }
    
    debugFile.close();
}
//
void NeuralNetwork::test(const vector<LabelData>& lData)
{
    cost   =0.;
    errRate=0.;
    
    auto totalISize = lData.size();
    auto nbBatches  = (totalISize-1)/Optim.batchSize + 1;
    
    for (size_t batch=0; batch<nbBatches; ++batch)
    {
        auto start  = batch*Optim.batchSize;
        auto end    = min(start+Optim.batchSize, lData.size());
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