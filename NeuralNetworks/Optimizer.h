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
    
    virtual ~Optimizer() {}
    virtual unique_ptr<Optimizer> clone() const = 0;
    
    auto getLambda()    const {return lambda;}
    auto getBatchSize() const {return batchSize;}
    auto getNbEpochs()  const {return nbEpochs;}
    
    virtual string getName()   const = 0;
    virtual string getDetail() const
    {
        return to_string(lambdaBase) + "_" + to_string(batchSize) + "_" + to_string(nbEpochs);
    }
    
    virtual void resize(size_t) = 0;
    virtual void updateParams(vec_r& params, const vec_r& dparams) = 0;
    
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
        alpha(_alpha)
    {};
    
    unique_ptr<Optimizer> clone() const override {return make_unique<GDOptimizer>(*this);}
    //
    string getName()   const override {return "GDOptim_" + getDetail() + "_" + Optimizer::getDetail();}
    string getDetail() const override {return to_string(alpha); }
    //
    void resize(size_t) override {}
    //
    void updateParams(vec_r& params, const vec_r& dparams) override
    {
        transform(params.begin(), params.end(), dparams.begin(), params.begin(), [a=alpha] (auto p, auto dp) {return p-a*dp;});
    }
    
protected:
    // members
    real alpha;
};
//
    
//
class NMOptimizer : public Optimizer
{
public:
    // methods
    NMOptimizer(real _alpha, real friction, real lambda, size_t batchSize, size_t nbEpochs, size_t trainSetSize) :
        Optimizer(lambda, batchSize, nbEpochs, trainSetSize),
        alpha(_alpha),
        friction(friction)
    {};
    
    unique_ptr<Optimizer> clone() const override { return make_unique<NMOptimizer>(*this); }
    //
    string getName()   const override { return "NMOptim_" + getDetail() + "_" + Optimizer::getDetail(); }
    string getDetail() const override { return to_string(alpha) + "_" + to_string(friction); }
    //
    void resize(size_t size) override
    {
        vparams.resize(size);
    }
    //
    void updateParams(vec_r& params, const vec_r& dparams) override
    {
        for (size_t o=0; o<params.size(); ++o)
        {
            auto pv   = vparams[o];
            auto nv   = friction*pv - alpha*dparams[o];

            params [o] += nv + friction*(nv-pv);
            vparams[o]  = nv;
        }
    }
    
protected:
    // members
    real alpha;
    real friction;
    
    vec_r vparams;
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
    
    unique_ptr<Optimizer> clone() const override {return make_unique<ADADOptimizer>(*this);}
    //
    string getName()   const override {return "ADADOptim_" + getDetail() + "_" + Optimizer::getDetail();}
    string getDetail() const override {return to_string(friction) + "_" + to_string(eps);}
    //
    void resize(size_t size) override
    {
        vparams.resize(size);
        xparams.resize(size);
    }
    //
    void updateParams(vec_r& params, const vec_r& dparams) override
    {
        for (size_t o=0; o<params.size(); ++o)
        {
            auto grad  = dparams[o];
            vparams[o] = friction*vparams[o] + (1-friction)*grad*grad;
            
            auto dp = -sqrt((xparams[o]+eps)/(vparams[o]+eps))*grad;
            
            params [o] += dp;
            xparams[o]  = friction*xparams[o] + (1-friction)*dp*dp;
        }
    }

protected:
    real eps;
    real friction;
    
    vec_r vparams;
    vec_r xparams;
};
    
}

#endif
