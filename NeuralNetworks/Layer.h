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
#include "Optimizer.h"

namespace NN {

enum class Phase { TRAIN, TEST};
enum class LayerClass { FCLayer, ConvLayer, ConvPoolLayer };
//
    
//
class Layer
{
protected:
    class LayerParams
    {
    private:
        class innerData
        {
        public:
            innerData() {}
            innerData(vec_r::iterator s, vec_r::iterator e) : s(s), e(e) {}
            
            auto begin()        {return s;}
            auto cbegin() const {return s;}
            auto end()          {return e;}
            auto cend()   const {return e;}
            
            auto&       operator[](size_t i)       {return *(s+i);}
            const auto& operator[](size_t i) const {return *(s+i);}
            
        private:
            vec_r::iterator s;
            vec_r::iterator e;
        };
    
    public:
        explicit LayerParams(size_t nbBias=0, size_t nbWeight=0);
        void     resize     (size_t nbBias  , size_t nbWeight  );
        void     reset() { fill(params.begin(), params.end(), 0.); }
        auto     size() const {return params.size();}
        
        auto begin()        {return params.begin();}
        auto cbegin() const {return params.cbegin();}
        auto end()          {return params.end();}
        auto cend()   const {return params.cend();}
        
        auto&       operator[](size_t i)       {return params[i];}
        const auto& operator[](size_t i) const {return params[i];}
        
        //
        vec_r     params;
        innerData weight;
        innerData bias;
    };

public:
    Layer(size_t size, real dropRate, const ActivationFunc& AFunc);
    ~Layer();
    
    // getters
    virtual string     getName()    const = 0;
    virtual string     getDetails() const = 0;
    virtual LayerClass getClass()   const = 0;
    
    auto        getInputSize () const {return inputSize;}
    auto        getOutputSize() const {return outputSize;}
    auto&       getA         ()       {return a;}
    const auto& getA         () const {return a;}
    const auto& getDrop      () const {return drop;}
    auto&       getDelta     ()       {return delta;}
    auto&       getParams    ()       {return params;}
    const auto& getDParams   () const {return dparams;}
    const auto& getAFunc     () const {return AFunc;}

    // setters
    void setNbData(size_t nbData) { resize(nbData);}
    void setPhase (Phase  p)      { phase = p; }
    void genDrop  ();
    
    // main methods
    virtual void setFromPrev(const Layer* prevLayer) = 0;
    virtual void fwdProp    (const Layer* prevLayer) = 0;
    virtual void bwdProp    (      Layer* prevLayer) = 0;
    virtual void calcGrad   (const Layer* prevLayer) = 0;
    
    void regularize  (real lambda);
    void updateParams(Optimizer& optim);
    
protected:
    static size_t layerCount;
    size_t        layerNb;
    size_t        inputSize;
    size_t        outputSize;
    size_t        weightInputSize;
    size_t        nbData;
    real          dropRate;
    Phase         phase;
    
    vec_r a;
    vec_r delta;
    vec_r drop;
    
    LayerParams  params; //bias and weight
    LayerParams dparams; //dbias and dweight

    const ActivationFunc& AFunc;
    default_random_engine gen;
    
    void initParams();
    virtual void resize(size_t nbData);
};
}

#endif /* defined(__NeuralNetworks__Layer__) */
