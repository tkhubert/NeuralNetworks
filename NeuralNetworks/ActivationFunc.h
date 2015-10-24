//
//  ActivationFunc.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef NeuralNetworks_ActivationFunc_h
#define NeuralNetworks_ActivationFunc_h

#include "NN.h"

namespace NN {
    
class ActivationFunc
{
public:
    ActivationFunc() {}
    virtual ~ActivationFunc() {}
    
    virtual string getName() const = 0;
    virtual real  f (real x) const = 0;
    virtual real  df(real f) const = 0;
};
//
    
//
class IdFunc: public ActivationFunc
{
public:
    IdFunc() {}
    
    string getName() const {return "IdFunc";}
    real f (real x)  const {return x;}
    real df(real f)  const {return 1;}
};
//
    
//
class SigmoidFunc : public ActivationFunc
{
public:
    SigmoidFunc() {}
    
    string getName() const {return "SigAFunc";}
    real f (real x)  const {return 1./(1+exp(-x));}
    real df(real f)  const {return f*(1-f);}
};
//
    
//
class TanHFunc : public ActivationFunc
{
public:
    TanHFunc() {}
    
    string getName() const {return "TanHAFunc";}
    real f (real x)  const
    {
        real tmp1 = exp(x);
        real tmp2 = 1/tmp1;
        return (tmp1-tmp2)/(tmp1+tmp2);
    }
    real df(real f) const {return 1.-f*f;}
};
//
    
//
class RLFunc : public ActivationFunc
{
public:
    RLFunc(real _a=1., real _b=0.001) : a(_a), b(_b) {}

    string getName() const {return "RLAFunc";}
    real f (real x) const {return x>=0. ? a*x : b*x;}
    real df(real f) const {return f>=0. ? a   : b;}
    
private:
    real a;
    real b;
};
    
}
#endif
