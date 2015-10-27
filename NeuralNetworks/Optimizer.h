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
    
    virtual void resize(const vec_i& sizes) = 0;
    virtual void updateParams(int idx, vec_r& params, const vec_r& dparams) = 0;
    
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
    
    string getName() const override
    {
        stringstream ss;
        ss << "GDOptim_" << getDetail() << "_" << Optimizer::getDetail();
        return ss.str();
    }
    //
    string getDetail() const override
    {
        stringstream ss;
        ss << alphaBase;
        return ss.str();
    }
    //
    void resize(const vec_i& sizes) override {}
    //
    void updateParams(int idx, vec_r& params, const vec_r& dparams) override
    {
        transform(params.begin(), params.end(), dparams.begin(), params.begin(), [a=alpha] (auto p, auto dp) {return p-a*dp;});
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
    
    string getName() const override
    {
        stringstream ss;
        ss << "NMOptim_" << getDetail() << "_" << Optimizer::getDetail();
        return ss.str();
    }
    //
    string getDetail() const override
    {
        stringstream ss;
        ss << alphaBase << "_" << friction;
        return ss.str();
    }
    //
    void resize(const vec_i& sizes) override
    {
        vvparams.resize(sizes.size());
        for (size_t i=0; i<sizes.size(); ++i)
            vvparams[i].resize(sizes[i]);
    }
    //
    void updateParams(int idx, vec_r& params, const vec_r& dparams) override
    {
        auto& vparams = vvparams[idx];
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
    real alpha, alphaBase;
    real friction;
    
    vector<vec_r> vvparams;
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
    
    string getName() const override
    {
        stringstream ss;
        ss << "ADADOptim_" << getDetail() << "_" << Optimizer::getDetail();
        return ss.str();
    }
    //
    string getDetail() const override
    {
        stringstream ss;
        ss << friction << "_" << eps;
        return ss.str();
    }
    //
    void resize(const vec_i& sizes) override
    {
        vvparams.resize(sizes.size());
        vxparams.resize(sizes.size());
        
        for (size_t i=0; i<sizes.size(); ++i)
        {
            vvparams[i].resize(sizes[i]);
            vxparams[i].resize(sizes[i]);
        }
    }
    //
    void updateParams(int idx, vec_r& params, const vec_r& dparams) override 
    {
        auto& vparams = vvparams[idx];
        auto& xparams = vxparams[idx];
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
    
    vector<vec_r> vvparams;
    vector<vec_r> vxparams;
};
    
}

#endif
