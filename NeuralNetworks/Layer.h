//
//  Layer.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef __NeuralNetworks__Layer__
#define __NeuralNetworks__Layer__

#include "NN.h"
#include "ActivationFunc.h"

namespace NN {

enum class Phase { TRAIN, TEST};
enum class LayerClass { FCLayer, ConvLayer, ConvPoolLayer };
//
class Layer
{
protected:
    struct Params
    {
        vec_r  params;
        size_t nbData;
        size_t nbBias;
        size_t nbWeight;
        size_t weightInputSize;
        
        explicit Params(size_t nbBias=0, size_t nbWeight=0, size_t weightInputSize=1);
        void     resize(size_t _nbBias , size_t _nbWeight , size_t _weightInputSize );
        void     reset() { fill(params.begin(), params.end(), 0.); }
        //
        const auto* const getCBPtr()    const {return &params[0];}
        const auto* const getCWPtr()    const {return &params[nbBias];}
        auto*             getBPtr()           {return &params[0];}
        auto*             getWPtr()           {return &params[nbBias];}
        //
    };

public:
    Layer(size_t size, real dropRate, const ActivationFunc& AFunc);
    ~Layer();
    
    virtual string     getName()    const = 0;
    virtual string     getDetails() const = 0;
    virtual LayerClass getClass()   const = 0;
    
    auto        getInputSize () const {return inputSize;}
    auto        getOutputSize() const {return outputSize;}
    auto        getLayerNb   () const {return layerNb;}
    const auto& getA         () const {return a;}
    const auto& getDrop      () const {return drop;}
    auto&       getDelta     ()       {return delta;}
    auto&       getParams    ()       {return params;}
    const auto& getParams    () const {return params;}
    const auto& getDParams   () const {return dparams;}

    const Layer*          getNextLayer() const {return nextLayer;}
    const Layer*          getPrevLayer() const {return prevLayer;}
    const ActivationFunc& getAFunc()     const {return AFunc;}

    void setNbData   (size_t nbData)          { resize(nbData);}
    void setNextLayer(Layer* next)             { nextLayer = next; }
    virtual void setPrevLayer(Layer* prev) = 0;
    void setPhase    (Phase  p)                { phase = p; }
    void setA        (const vec_r& _a) { a = _a;}
    void setA        (vec_r&&      _a) { a = move(_a);}
    void setDrop     ();
    
    virtual void setDCost(const vec_r& dc);
    virtual void fwdProp()  = 0;
    virtual void bwdProp()  = 0;
    virtual void calcGrad() = 0;
    virtual void regularize(real lambda);

protected:
    static size_t layerCount;
    size_t        layerNb;
    size_t        inputSize;
    size_t        outputSize;
    size_t        nbData;
    real          dropRate;
    Phase         phase;
    
    vec_r a;
    vec_r delta;
    vec_r drop;
    
    Params  params; //bias and weight
    Params dparams; //dbias and dweight

    Layer* nextLayer;
    Layer* prevLayer;

    const ActivationFunc& AFunc;
    default_random_engine gen;
    
    void initParams();
    virtual void resize(size_t nbData);
};
}

#endif /* defined(__NeuralNetworks__Layer__) */
