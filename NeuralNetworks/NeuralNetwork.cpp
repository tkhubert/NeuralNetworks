//
//  NeuralNetwork.cpp
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#include "NeuralNetwork.h"

namespace NN
{
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
        
        cLayer->setFromPrev(pLayer.get());
    }
    
    inputSize  = layers.front()->getOutputSize();
    outputSize = layers.back()->getOutputSize();
    
}
//
string NeuralNetwork::getName() const
{
    string s;
    for (size_t i=0; i<nbLayers; ++i)
        s+= to_string(layers[i]->getOutputSize()) + "_";

    s += CFunc.getName() + "_" + layers.back()->getAFunc().getName();
    return s;
}
//
void NeuralNetwork::setNbData(size_t nbData)
{
    for_each(layers.begin(), layers.end(), [nbData] (auto& l) { l->setNbData(nbData);});
}
//
void NeuralNetwork::setPhase(Phase phase)
{
    for_each(layers.begin(), layers.end(), [phase] (auto& l) { l->setPhase(phase);});
}
//
void NeuralNetwork::genDrop()
{
    for_each(layers.begin(), layers.end(), [] (auto& l) { l->genDrop();});
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
    
    auto&          a = layers.front()->getA();
    const auto& drop = layers.front()->getDrop();
    transform(input.begin(), input.end(), drop.begin(), a.begin(), [] (auto inp, auto d) {return inp*d;});
}
//
    
//
void NeuralNetwork::fwdProp(LabelDataCItr dataStart, LabelDataCItr dataEnd)
{
    genDrop();
    setInput(dataStart, dataEnd);
    for (size_t i=1; i<nbLayers; ++i)
        layers[i]->fwdProp(layers[i-1].get());
}
//
void NeuralNetwork::bwdProp(LabelDataCItr dataStart, LabelDataCItr dataEnd)
{
    setDCost(dataStart, dataEnd);
    for (size_t i=nbLayers-1; i>=2; --i)
        layers[i]->bwdProp(layers[i-1].get());
}
//
void NeuralNetwork::calcGrad()
{
    for (size_t i=1; i<nbLayers; ++i)
        layers[i]->calcGrad(layers[i-1].get());
}
//
void NeuralNetwork::regularize(real lambda)
{
    for_each(layers.begin()+1, layers.end(), [lambda] (auto& l) { l->regularize(lambda);});
}
//
void NeuralNetwork::updateParams(vector<unique_ptr<Optimizer>>& optims)
{
    for (size_t i=1; i<nbLayers; ++i)
        layers[i]->updateParams(*optims[i]);
}
//
    
//
void NeuralNetwork::setDCost(LabelDataCItr dataStart, LabelDataCItr dataEnd)
{
    vec_r dc = CFunc.df(getOutput(), dataStart, dataEnd);
    
    const auto& AFunc = layers.back()->getAFunc();
    const auto& a     = layers.back()->getA();
    auto&       delta = layers.back()->getDelta();
    
    transform(a.begin(), a.end(), dc.begin(), delta.begin(), [&AFunc] (auto a, auto dc) {return AFunc.df(a)*dc;});
}
//
real NeuralNetwork::calcCost(LabelDataCItr dataStart, LabelDataCItr dataEnd) const
{
    return CFunc.f(getOutput(), dataStart, dataEnd);
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
void NeuralNetwork::train(const DataContainer& data, const Optimizer& optim)
{
    auto name = getName() + "_" + optim.getName();
    ofstream debugFile  = ofstream("/Users/tkhubert/Documents/Projects/NeuralNetworks/MNist/"+name+".csv");

    cout << "Start training " << name << "-------------"<<endl;
    
    vector<unique_ptr<Optimizer>> optims(layers.size());
    for (size_t i=0; i<layers.size(); ++i)
    {
        optims[i] = optim.clone();
        optims[i]->resize(layers[i]->getParams().size());
    }
    
    auto lData      = data.getTrainLabelData();
    auto nbEpochs   = optim.getNbEpochs();
    auto batchSize  = optim.getBatchSize();
    auto totalISize = lData.size();
    auto nbBatches  = (totalISize-1)/batchSize + 1;
    
    for (size_t t=0; t<nbEpochs; ++t)
    {
        setPhase(Phase::TRAIN);
        
        debugFile << t << ", ";
        cout << "Epoch: " << t << ", ";
        clock_t startTimeEpoch = clock();
        
        random_shuffle(lData.begin(), lData.end());
        
        for (size_t batch=0; batch<nbBatches; ++batch)
        {
            auto start  = batch*batchSize;
            auto end    = min(start+batchSize, lData.size());
            auto nbData = end-start;
            
            auto dataStart = lData.cbegin()+start;
            auto dataEnd   = dataStart+nbData;
            setNbData(nbData);
            
            fwdProp (dataStart, dataEnd);
            bwdProp (dataStart, dataEnd);
            calcGrad();
            
            regularize(optim.getLambda());
            updateParams(optims);
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
        
        debugFile << "time "      << timeEpoch << "s,";
        debugFile << trainErrRate << "," << crossErrRate << "," << testErrRate << ",";
        debugFile << trainCost    << "," << crossCost    << "," << testCost    << endl;
        cout      << "time "      << timeEpoch << "s,";
        cout      << trainErrRate << "," << crossErrRate << "," << testErrRate << ",";
        cout      << trainCost    << "," << crossCost    << "," << testCost    << endl;
        
        if (CHECKGRAD)
        {
            default_random_engine gen;
            uniform_int_distribution<int> unif(0, nbBatches);
            auto batch = unif(gen);
            
            auto start  = batch*batchSize;
            auto end    = min(start+batchSize, lData.size());
            auto nbData = end-start;
            
            auto dataStart = lData.cbegin()+start;
            auto dataEnd   = dataStart+nbData;
            
            checkGradient(dataStart, dataEnd);
        }
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
        setNbData(nbData);
        
        fwdProp(dataStart, dataEnd);
        cost    += calcCost (dataStart, dataEnd);
        errRate += isCorrect(dataStart, dataEnd);
    }
    
    cost    /= totalISize;
    errRate  = 1.-errRate/totalISize;
}
//
void NeuralNetwork::checkGradient(LabelDataCItr lDStart, LabelDataCItr lDEnd)
{
    cout << "  Checking Gradient --------" << endl;

    vec_r dC(outputSize);
    
    auto nbData = distance(lDStart, lDEnd);
    setNbData(nbData);
    
    fwdProp  (lDStart, lDEnd);
    bwdProp  (lDStart, lDEnd);
    calcGrad ();
    
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
            auto dp  = TWEAKSIZE*p;
            
            params[idx] = p+dp;
            fwdProp(lDStart, lDEnd);
            auto cu = calcCost(lDStart, lDEnd);
            
            params[idx] = p-dp;
            fwdProp(lDStart, lDEnd);
            auto cd = calcCost(lDStart, lDEnd);
            
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
                cout << "    " << i << " " << n << " " << idx << " " << grad << " " << deriv << " " << err << endl;
            }
            
            params[idx] = p;
        }
        cout << "   Gradient of layer " << i;
        if (pass) cout << " is correct"   << endl;
        else      cout << " is incorrect" << endl;
            
    }
    
}
//
}