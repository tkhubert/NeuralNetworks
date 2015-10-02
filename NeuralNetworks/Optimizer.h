//
//  Optimizer.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 09/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef NeuralNetworks_Optimizer_h
#define NeuralNetworks_Optimizer_h

#include "NN.h"
#include "Layer.h"

namespace NN {

class Optimizer
{
public:
    // methods
    Optimizer(float _lambda, size_t batchSize, size_t nbEpochs, size_t trainSetSize) :
        lambda(_lambda*batchSize/trainSetSize),
        batchSize(batchSize),
        nbEpochs(nbEpochs),
        lambdaBase(_lambda)
    {};
    
    auto getBatchSize() const {return batchSize;}
    auto getNbEpochs()  const {return nbEpochs;}
    
    virtual string getName()   const = 0;
    virtual string getDetail() const
    {
        stringstream ss;
        ss << lambdaBase << "_" << batchSize << "_" << nbEpochs;
        return ss.str();
    }
    
    virtual void resize(const vector<unique_ptr<Layer>>& net) = 0;
    virtual void updateParams(Layer& layer) = 0;
    
protected:
    float  lambda, lambdaBase;
    size_t batchSize;
    size_t nbEpochs;
};
//

//
class GDOptimizer : public Optimizer
{
public:
    // methods
    GDOptimizer(float _alpha, float lambda, size_t batchSize, size_t nbEpochs, size_t trainSetSize) :
        Optimizer(lambda, batchSize, nbEpochs, trainSetSize),
        alpha(_alpha/batchSize),
        alphaBase(_alpha)
    {};
    
    string getName() const
    {
        stringstream ss;
        ss << "GDOptim_" << getDetail() << "_" << Optimizer::getDetail();
        return ss.str();
    }
    //
    string getDetail() const
    {
        stringstream ss;
        ss << alphaBase;
        return ss.str();
    }
    //
    void resize(const vector<unique_ptr<Layer>>& net) {}
    //
    void updateParams(Layer& layer)
    {
        auto& bias    = layer.getBias();
        auto& dbias   = layer.getDBias();
        auto& weight  = layer.getWeight();
        auto& dweight = layer.getDWeight();
        
        transform(bias.begin()  , bias.end()  , dbias.begin()  , bias.begin()  , [a=alpha]           (auto b, auto db) {return b-a*db;      });
        transform(weight.begin(), weight.end(), dweight.begin(), weight.begin(), [a=alpha, l=lambda] (auto w, auto dw) {return w-a*(dw+l*w);});
        fill(dbias.begin()  , dbias.end()  , 0.);
        fill(dweight.begin(), dweight.end(), 0.);
    }
    
protected:
    // members
    float  alpha, alphaBase;
};
//
    
//
class NMOptimizer : public Optimizer
{
public:
    // methods
    NMOptimizer(float _alpha, float friction, float lambda, size_t batchSize, size_t nbEpochs, size_t trainSetSize) :
        Optimizer(lambda, batchSize, nbEpochs, trainSetSize),
        alpha(_alpha/batchSize),
        friction(friction),
        alphaBase(_alpha)
    {};
    
    string getName() const
    {
        stringstream ss;
        ss << "NMOptim_" << getDetail() << "_" << Optimizer::getDetail();
        return ss.str();
    }
    //
    string getDetail() const
    {
        stringstream ss;
        ss << alphaBase << "_" << friction;
        return ss.str();
    }
    //
    void resize(const vector<unique_ptr<Layer>>& net)
    {
        auto nbLayers = net.size();
        vvbias.resize(nbLayers);
        vvweight.resize(nbLayers);
        
        for (size_t i=0; i<nbLayers; ++i)
        {
            vvbias[i].resize(net[i]->getBias().size());
            vvweight[i].resize(net[i]->getWeight().size());
        }
    }
    //
    void updateParams(Layer& layer)
    {
        auto  layerNb = layer.getLayerNb();
        auto& bias    = layer.getBias();
        auto& dbias   = layer.getDBias();
        auto& weight  = layer.getWeight();
        auto& dweight = layer.getDWeight();
        
        auto& vbias = vvbias[layerNb];
        for (size_t o=0; o<bias.size(); ++o)
        {
            auto pv   = vbias[o];
            auto nv   = friction*pv - alpha*dbias[o];

            bias [o] += nv + friction*(nv-pv);
            dbias[o]  = 0.;
            vbias[o]  = nv;
        }
        
        auto& vweight = vvweight[layerNb];
        for (size_t o=0; o<weight.size(); ++o)
        {
            auto pv     = vweight[o];
            auto nv     = friction*pv - alpha*(dweight[o]+lambda*weight[o]);
            
            weight [o] += nv + friction*(nv-pv);
            dweight[o]  = 0.;
            vweight[o]  = nv;
        }
    }
    
protected:
    // members
    float  alpha, alphaBase;
    float  friction;
    
    vector<vector<float>> vvbias;
    vector<vector<float>> vvweight;
};
//
    
}

#endif
