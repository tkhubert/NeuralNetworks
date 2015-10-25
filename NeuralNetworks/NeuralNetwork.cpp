//
//  NeuralNetwork.cpp
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#include "NeuralNetwork.h"

namespace NN {
    
constexpr auto NBGRADTEST = 10;
constexpr auto GRADTOL    = 1e-6;
//
    
//
NeuralNetwork::NeuralNetwork(const CostFunc& CFunc, vector<unique_ptr<Layer>>&& _layers) :
    nbLayers(_layers.size()),
    CFunc(CFunc),
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
        optim.updateParams(i, layers[i]->getParams().params, layers[i]->getDParams().params);
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
    
    vec_r input(nbData*dataSize);
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
void NeuralNetwork::bwdProp(const vec_r& dC)
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
void NeuralNetwork::regularize(real lambda)
{
    for (size_t i=1; i<nbLayers; ++i)
        layers[i]->regularize(lambda);
}
//
real NeuralNetwork::calcCost(LabelDataCItr dataStart, LabelDataCItr dataEnd) const
{
    return CFunc.f(getOutput(), dataStart, dataEnd);
}
//
void NeuralNetwork::calcDCost(LabelDataCItr dataStart, LabelDataCItr dataEnd, vec_r& dC)
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
            
            vec_r dC(outputSize*nbData);
            auto dataStart = lData.cbegin()+start;
            auto dataEnd   = dataStart+nbData;
            
            fwdProp  (dataStart, dataEnd);
            calcDCost(dataStart, dataEnd, dC);
            bwdProp  (dC);
            calcGrad ();
            regularize(optim.getLambda());
            
            updateParams(optim);
        }
        
        test(data.getTrainLabelData(), batchSize);
        auto trainErrRate = errRate;
        auto trainCost    = cost;
        
        test(data.getCrossLabelData(), batchSize);
        auto crossErrRate = errRate;
        auto crossCost    = cost;
        
        test(data.getTestLabelData(), batchSize);
        auto testErrRate = errRate;
        auto testCost    = cost;
        
        auto timeEpoch = ( clock() - startTimeEpoch ) / (real) CLOCKS_PER_SEC;
        debugFile << "time " << timeEpoch << "s,";
        cout << "time " << timeEpoch << "s,";
        
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
//
void NeuralNetwork::checkGradient(const LabelData& lD)
{
    cout << "Checking Gradient --------" << endl;

    vec_r    dC(outputSize);
    vector<LabelData> labelData(1);
    
    labelData[0] = lD;
    fwdProp(labelData.cbegin(), labelData.cend());
    calcDCost(labelData.cbegin(), labelData.cend(), dC);
    bwdProp  (dC);
    calcGrad ();
    
    default_random_engine gen;
    
    for (size_t i=1; i<nbLayers; ++i)
    {
        auto& params  = layers[i]->getParams().params;
        auto& dparams = layers[i]->getDParams().params;
        
        vector<pair<int,real>> dpparams(dparams.size());
        for (size_t j=0; j<dparams.size(); ++j)
            dpparams[j] = make_pair(j, fabs(dparams[j]));
        sort(dpparams.begin(), dpparams.end(), [] (auto e1, auto e2) {return e2.second < e1.second;});

        bool pass = true;
        auto nbTest = params.size()>0 ? NBGRADTEST : 0;
        
        for (size_t n=0; n<nbTest; ++n)
        {
            auto idx = dpparams[n].first;
            auto p   = params[idx];
            auto dp  = 1e-2*p;
            
            params[idx] = p+dp;
            fwdProp(labelData.cbegin(), labelData.cend());
            auto cu = calcCost(labelData.cbegin(), labelData.cend());
            
            params[idx] = p-dp;
            fwdProp(labelData.cbegin(), labelData.cend());
            auto cd = calcCost(labelData.cbegin(), labelData.cend());
            
            auto deriv = (cu-cd)/(2*dp);
            auto grad  = dparams[idx];
            
            auto err = 1.;
            if (fabs(grad)<TINY && abs(deriv)<TINY)
                err = 0;
            else if (fabs(deriv)>TINY)
                err = fabs((deriv-grad)/deriv);
            else
                err = fabs((deriv-grad)/grad);
            
            if (err>GRADTOL)
            {
                pass = false;
                cout << i << " " << n << " " << idx << " " << grad << " " << deriv << " " << err << endl;
            }
            
            params[idx] = p;
        }
        cout << "Gradient of layer " << i;
        if (pass) cout << " is correct"   << endl;
        else      cout << " is incorrect" << endl;
            
    }
    
}
//
}