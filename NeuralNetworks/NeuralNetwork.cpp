//
//  NeuralNetwork.cpp
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const CostFunc& _CFunc, const Optimizer& _Optim, std::vector<Layer*>& _layers) : nbLayers(0) , CFunc(_CFunc), Optim(_Optim)
{
    nbLayers = _layers.size();
    layers.resize(nbLayers);
    
    layers[0] = _layers[0];
    for (size_t i=1; i<nbLayers; ++i)
    {
        Layer* pLayer = layers[i-1];
        Layer* cLayer = _layers[i];
        assert(pLayer->getOutputSize()==cLayer->getInputSize());
        
        pLayer->setNextLayer(cLayer);
        cLayer->setPrevLayer(pLayer);
        layers[i] = cLayer;
    }
    
    inputSize  = layers[0]->getOutputSize();
    outputSize = layers[nbLayers-1]->getOutputSize();
    
    debugFile  = std::ofstream("/Users/tkhubert/Documents/Projects/NeuralNetworks/MNist/"+getName());
}
//
std::string NeuralNetwork::getName() const
{
    std::stringstream ss;
    for (size_t i=0; i<nbLayers; ++i)
        ss << layers[i]->getOutputSize() << "_";

    ss << CFunc.getName() << "_" << layers[nbLayers-1]->getAFunc().getName() << "_" << Optim.getName();
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
    layers[0]->setA(lD.data);
}
//
void NeuralNetwork::setInput(std::vector<LabelData>::const_iterator dataStart, std::vector<LabelData>::const_iterator dataEnd)
{
    auto nbData   = std::distance(dataStart, dataEnd);
    auto dataSize = dataStart->data.size();
    std::vector<float> input(nbData*dataSize);
    
    for (size_t d=0; d<nbData; ++d)
    {
        const LabelData& lD = *(dataStart+d);
        
        for (size_t i=0; i<dataSize; ++i)
            input[d*inputSize+i] = lD.data[i];
    }
    
    setNbData(nbData);
    layers[0]->setA(input);
}
//
void NeuralNetwork::fwdProp(const LabelData& lD)
{
    setInput(lD);
    for (size_t i=1; i<nbLayers; ++i)
        layers[i]->fwdProp();
}
//
void NeuralNetwork::fwdProp(std::vector<LabelData>::const_iterator dataStart, std::vector<LabelData>::const_iterator dataEnd)
{
    setInput(dataStart, dataEnd);
    for (size_t i=1; i<nbLayers; ++i)
        layers[i]->fwdProp();
}
//
void NeuralNetwork::bwdProp(const std::vector<float>& dC)
{
    setDCost(dC);
    for (size_t i=nbLayers-1; i>=1; --i)
        layers[i]->bwdProp();
}
//
float NeuralNetwork::calcCost(std::vector<LabelData>::const_iterator dataStart, std::vector<LabelData>::const_iterator dataEnd) const
{
    return CFunc.f(getOutput(), dataStart, dataEnd);
}
//
void NeuralNetwork::calcDCost(std::vector<LabelData>::const_iterator dataStart, std::vector<LabelData>::const_iterator dataEnd, std::vector<float>& dC)
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
size_t NeuralNetwork::isCorrect(std::vector<LabelData>::const_iterator dataStart, std::vector<LabelData>::const_iterator dataEnd) const
{
    const auto& prediction = getOutput();
    auto nbData = std::distance(dataStart, dataEnd);
    
    size_t nbCorrect = 0;
    for (size_t d=0; d<nbData; ++d)
    {
        auto s = prediction.cbegin()+d*outputSize;
        auto e = s + outputSize;
        const LabelData& lD = *(dataStart+d);
        
        nbCorrect += std::distance(s, std::max_element(s, e))==lD.label;
    }
    return nbCorrect;
}
//
void NeuralNetwork::train(const DataContainer& data)
{
    auto lData = data.getTrainLabelData();
    
    std::cout << "Start training " << getName() << "-------------"<<std::endl;
    
    auto totalISize = lData.size();
    auto nbBatches  = (totalISize-1)/Optim.batchSize + 1;
    
    for (size_t t=0; t<Optim.nbEpochs; ++t)
    {
        debugFile << t << ", ";
        std::cout << "Epoch: " << t << ", ";
        std::clock_t startTimeEpoch = std::clock();
        
        std::random_shuffle(lData.begin(), lData.end());
        
        for (size_t batch=0; batch<nbBatches; ++batch)
        {
            size_t start  = batch*Optim.batchSize;
            size_t end    = std::min(start+Optim.batchSize, lData.size());
            size_t nbData = end-start;
            
            std::vector<float> dC(outputSize*nbData);
            auto dataStart = lData.cbegin()+start;
            auto dataEnd   = dataStart+nbData;
            
            fwdProp  (dataStart, dataEnd);
            calcDCost(dataStart, dataEnd, dC);
            bwdProp  (dC);
            
            updateParams();
        }
        
        auto timeEpoch = ( std::clock() - startTimeEpoch ) / (float) CLOCKS_PER_SEC;
        debugFile << "time " << timeEpoch << "s,";
        std::cout << "time " << timeEpoch << "s,";
        
        test(data.getTrainLabelData());
        float trainErrRate = errRate;
        float trainCost    = cost;
        
        test(data.getCrossLabelData());
        float crossErrRate = errRate;
        float crossCost    = cost;
        
        test(data.getTestLabelData());
        float testErrRate = errRate;
        float testCost    = cost;
        
        debugFile << trainErrRate << "," << crossErrRate << "," << testErrRate << ",";
        debugFile << trainCost    << "," << crossCost    << "," << testCost    << std::endl;
        std::cout << trainErrRate << "," << crossErrRate << "," << testErrRate << ",";
        std::cout << trainCost    << "," << crossCost    << "," << testCost    << std::endl;
    }
    
    debugFile.close();
}
//
void NeuralNetwork::test(const std::vector<LabelData>& lData)
{
    cost   =0.;
    errRate=0.;
    
    auto totalISize = lData.size();
    auto nbBatches  = (totalISize-1)/Optim.batchSize + 1;
    
    for (size_t batch=0; batch<nbBatches; ++batch)
    {
        auto start  = batch*Optim.batchSize;
        auto end    = std::min(start+Optim.batchSize, lData.size());
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