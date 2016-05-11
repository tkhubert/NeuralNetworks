//
//  Regularizer.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 11/05/2016.
//
//

#ifndef Regularizer_h
#define Regularizer_h

#include "NN.h"

namespace NN
{
    
class Regularizer
{
public:
    // methods
    Regularizer() {};
    virtual ~Regularizer() {}
    virtual unique_ptr<Regularizer> clone() const = 0;
    
    virtual string getName()   const = 0;
    virtual string getDetail() const = 0;
    
    virtual void apply(vecr_itr weightBegin, vecr_itr weightEnd, vecr_itr dweightBegin) const = 0;
};
//

//
class L2Regularizer : public Regularizer
{
public:
    L2Regularizer(real _lambda) : lambda(_lambda) {};
    
    unique_ptr<Regularizer> clone() const override {return make_unique<L2Regularizer>(*this);}
    //
    string getName()   const override {return "L2Reg_" + getDetail();}
    string getDetail() const override {return to_string(lambda); }
    //
    void apply(vecr_itr weightBegin, vecr_itr weightEnd, vecr_itr dweightBegin) const override
    {
        transform(weightBegin, weightEnd, dweightBegin, dweightBegin, [lambda=this->lambda] (auto w, auto dw) {return dw+lambda*w;});
    }
    
private:
    real lambda;
    // lambda = lambda * batchSize/trainSetSize!
};
    
}

#endif /* Regularizer_h */
