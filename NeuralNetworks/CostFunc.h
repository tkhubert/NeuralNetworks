//
//  CostFunc.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef NeuralNetworks_CostFunc_h
#define NeuralNetworks_CostFunc_h

#include "includes.h"

class CostFunc
{
public:
    CostFunc() {}
    virtual double  f(const std::vector<double>& a, int y) const = 0;
    virtual double df(size_t i, const std::vector<double>& a, int y) const = 0;
};
//
class MSECostFunc : public CostFunc
{
public:
    MSECostFunc() {}
    double  f(const std::vector<double>& a, int y) const
    {
        double val=0.;
        for (size_t i=0; i<a.size(); ++i)
        {
            int label = i==y;
            val+= (a[i]-label)*(a[i]-label);
        }
        return 0.5*val;
    }
    //
    double df(size_t i, const std::vector<double>& a, int y) const
    {
        int label = i==y;
        return a[i]-label;
    }
};
//
class CECostFunc : public CostFunc
{
public:
    CECostFunc() {}
    double  f(const std::vector<double>& a, int y) const
    {
        double val=0.;
        for (size_t i=0; i<a.size(); ++i)
        {
            int label = i==y;
            val += -label*std::log(a[i]) - (1-label)*std::log(1-a[i]);
        }
        return val;
    }
    //
    double df(size_t i, const std::vector<double>& a, int y, std::vector<double>& da) const
    {
        int label = i==y;
        return -(label-a[i])/(a[i]*(1-a[i]));
    }
};

#endif
