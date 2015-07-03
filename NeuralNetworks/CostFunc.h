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
    virtual std::string getName() const = 0;
    virtual double      f(const std::vector<double>& a, int y) const = 0;
    virtual void        df(const std::vector<double>& a, int y, std::vector<double>& dc) const = 0;
};
//
class MSECostFunc : public CostFunc
{
public:
    MSECostFunc() {}
    
    std::string getName() const {return "MSECFunc";}
    
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
    void df(const std::vector<double>& a, int y, std::vector<double>& dc) const
    {
        for (size_t i=0; i<a.size(); ++i)
        {
            int label = i==y;
            dc[i] = a[i]-label;
        }
    }
};
//
class CECostFunc : public CostFunc
{
public:
    CECostFunc() {}
    
    std::string getName() const {return "CECFunc";}
    
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
    void df(const std::vector<double>& a, int y, std::vector<double>& dc) const
    {
        for (size_t i=0; i<a.size(); ++i)
            dc[i] = 1/(1-a[i]);
        
        dc[y] = -1/a[y];
    }
};
//
class SMCostFunc : public CostFunc
{
public:
    SMCostFunc() {}
    
    std::string getName() const {return "SMCFunc";}
    
    double f(const std::vector<double>& a, int y) const
    {
        double maxA = *(std::max_element(a.begin(), a.end()));
        std::vector<double> expA(a.size());
        
        double sum = 0.;
        for (size_t i=0; i<a.size(); ++i)
        {
            expA[i] = exp(a[i]-maxA);
            sum += expA[i];
        }
        
        return -log(expA[y]/sum);
    }
    //
    void df(const std::vector<double>&a, int y, std::vector<double>& dc) const
    {
        std::vector<double> expA(a.size());
        
        double sum = 0.;
        for (size_t i=0; i<a.size(); ++i)
        {
            expA[i] = exp(a[i]);
            sum += expA[i];
        }
        
        for (size_t i=0; i<a.size(); ++i)
            dc[i] = expA[i]/sum;
        
        dc[y] -=1;
    }
};
//
class SVMCostFunc : public CostFunc
{
public:
    SVMCostFunc() {}
    
    std::string getName() const {return "SVMCFunc";}
    
    double f(const std::vector<double>& a, int y) const
    {
        double val = -1.;
        for (size_t i=0; i<a.size(); ++i)
            val += std::max(0.,a[i]-a[y]+1);
        return val;
    }
    //
    void df(const std::vector<double>&a, int y, std::vector<double>& dc) const
    {
        for (size_t i=0; i<a.size(); ++i)
            dc[i] = a[i]-a[y]>-1 ? 1. : 0.;

        for (size_t i=0; i<a.size(); ++i)
            dc[y] += a[i]-a[y]>-1 ? -1. : 0;
    }
};
//

#endif
