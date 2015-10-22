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
    Optimizer(real _lambda, size_t batchSize, size_t nbEpochs, size_t trainSetSize) :
        lambda(_lambda*batchSize/trainSetSize),
        lambdaBase(_lambda),
        batchSize(batchSize),
        nbEpochs(nbEpochs)
    {};
    
    auto getLambda()    const {return lambda;}
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
    real   lambda, lambdaBase;
    size_t batchSize;
    size_t nbEpochs;
};
//

//
class GDOptimizer : public Optimizer
{
public:
    // methods
    GDOptimizer(real _alpha, real lambda, size_t batchSize, size_t nbEpochs, size_t trainSetSize) :
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
        
        transform(bias.begin()  , bias.end()  , dbias.begin()  , bias.begin()  , [a=alpha] (auto b, auto db) {return b-a*db;});
        transform(weight.begin(), weight.end(), dweight.begin(), weight.begin(), [a=alpha] (auto w, auto dw) {return w-a*dw;});
    }
    
protected:
    // members
    real alpha, alphaBase;
};
//
    
//
class NMOptimizer : public Optimizer
{
public:
    // methods
    NMOptimizer(real _alpha, real friction, real lambda, size_t batchSize, size_t nbEpochs, size_t trainSetSize) :
        Optimizer(lambda, batchSize, nbEpochs, trainSetSize),
        alpha(_alpha/batchSize),
        alphaBase(_alpha),
        friction(friction)
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
            vbias[o]  = nv;
        }
        
        auto& vweight = vvweight[layerNb];
        for (size_t o=0; o<weight.size(); ++o)
        {
            auto pv     = vweight[o];
            auto nv     = friction*pv - alpha*dweight[o];
            
            weight [o] += nv + friction*(nv-pv);
            vweight[o]  = nv;
        }
    }
    
protected:
    // members
    real alpha, alphaBase;
    real friction;
    
    vector<vec_r> vvbias;
    vector<vec_r> vvweight;
};
//
    
//
class ADADOptimizer : public Optimizer
{
public:
    // methods
    ADADOptimizer(real friction, real eps, real lambda, size_t batchSize, size_t nbEpochs, size_t trainSetSize) :
        Optimizer(lambda, batchSize, nbEpochs, trainSetSize),
        eps(eps),
        friction(friction)
    {};
    
    string getName() const
    {
        stringstream ss;
        ss << "ADADOptim_" << getDetail() << "_" << Optimizer::getDetail();
        return ss.str();
    }
    //
    string getDetail() const
    {
        stringstream ss;
        ss << friction << "_" << eps;
        return ss.str();
    }
    //
    void resize(const vector<unique_ptr<Layer>>& net)
    {
        auto nbLayers = net.size();
        vvbias.resize(nbLayers);
        vxbias.resize(nbLayers);
        vvweight.resize(nbLayers);
        vxweight.resize(nbLayers);
        
        for (size_t i=0; i<nbLayers; ++i)
        {
            vvbias[i].resize(net[i]->getBias().size());
            vxbias[i].resize(net[i]->getBias().size());
            vvweight[i].resize(net[i]->getWeight().size());
            vxweight[i].resize(net[i]->getWeight().size());
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
        auto& xbias = vxbias[layerNb];
        for (size_t o=0; o<bias.size(); ++o)
        {
            auto grad  = dbias[o];
            vbias[o] = friction*vbias[o] + (1-friction)*grad*grad;
            
            auto db = -sqrt((xbias[o]+eps)/(vbias[o]+eps))*grad;
            
            bias [o] += db;
            xbias[o]  = friction*xbias[o] + (1-friction)*db*db;
        }
        
        auto& vweight = vvweight[layerNb];
        auto& xweight = vxweight[layerNb];
        for (size_t o=0; o<weight.size(); ++o)
        {
            auto grad = dweight[o];
            vweight[o] = friction*vweight[o] + (1-friction)*grad*grad;
            
            auto dw = -sqrt((xweight[o]+eps)/(vweight[o]+eps))*grad;
            
            weight [o] += dw;
            xweight[o]  = friction*xweight[o] + (1-friction)*dw*dw;
        }
    }

    
protected:
    real eps;
    real friction;
    
    vector<vec_r> vvbias;
    vector<vec_r> vvweight;
    vector<vec_r> vxbias;
    vector<vec_r> vxweight;
};
    
}

#endif
