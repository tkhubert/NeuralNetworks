//
//  ActivationFunc.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef NeuralNetworks_ActivationFunc_h
#define NeuralNetworks_ActivationFunc_h

#include "includes.h"

class ActivationFunc
{
public:
    ActivationFunc() {}
    virtual std::string getName()    const = 0;
    virtual double      f(double x)  const = 0;
    virtual double      df(double f) const = 0;
};
//
class SigmoidFunc : public ActivationFunc
{
public:
    SigmoidFunc() {}
    
    std::string getName() const {return "SigAFunc";}
    double f (double x)   const {return 1./(1+std::exp(-x));}
    double df(double f)   const {return f*(1-f);}
};
//
class TanHFunc : public ActivationFunc
{
public:
    TanHFunc() {}
    
    std::string getName() const {return "TanHAFunc";}
    double f (double x) const
    {
        double tmp1 = std::exp(x);
        double tmp2 = 1/tmp1;
        return (tmp1-tmp2)/(tmp1+tmp2);
    }
    double df(double f) const {return 1.-f*f;}
};
//
class RLFunc : public ActivationFunc
{
public:
    std::string getName() const {return "RLAFunc";}
    RLFunc(double _a=1.0, double _b=0.) : a(_a), b(_b) {}
    double f (double x) const {return a*std::max(x,0.)-b*std::max(-x,0.);}
    double df(double f) const {return f>=0 ? a : b;}
    
private:
    double a;
    double b;
};
#endif
