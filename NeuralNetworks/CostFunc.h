//
//  CostFunc.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef NeuralNetworks_CostFunc_h
#define NeuralNetworks_CostFunc_h

#include "NN.h"
#include "Data.h"

namespace NN {

using LabelDataCItr = vector<LabelData>::const_iterator;
    
class CostFunc
{
public:
    CostFunc() {}
    virtual string getName() const = 0;

    virtual float f(const vector<float>& a, LabelDataCItr dataStart, LabelDataCItr dataEnd) const = 0;
    virtual void df(const vector<float>& a, LabelDataCItr dataStart, LabelDataCItr dataEnd, vector<float>& dc) const = 0;
};
//
class MSECostFunc : public CostFunc
{
public:
    MSECostFunc() {}
    
    string getName() const {return "MSECFunc";}
    //
    float f(const vector<float>& a, LabelDataCItr dataStart, LabelDataCItr dataEnd) const
    {
        float val        = 0;
        auto  nbData     = distance(dataStart, dataEnd);
        auto  outputSize = a.size()/nbData;

        for (size_t d=0; d<nbData; ++d)
        {
            auto s = a.cbegin() + d*outputSize;
            auto y = (dataStart+d)->label;
            
            for (size_t i=0; i<outputSize; ++i)
            {
                auto label = i==y;
                auto aVal  = *(s+i);
                val+= 0.5*(aVal-label)*(aVal-label);
            }
        }
        return val;
    }
    //
    void df(const vector<float>& a, LabelDataCItr dataStart, LabelDataCItr dataEnd, vector<float>& dc) const
    {
        auto nbData     = distance(dataStart, dataEnd);
        auto outputSize = a.size()/nbData;
        
        vector<float> aloc(outputSize);
        vector<float> dcloc(outputSize);
        
        for (size_t d=0; d<nbData; ++d)
        {
            auto s = a.cbegin() + d*outputSize;
            auto y = (dataStart+d)->label;
            
            for (size_t i=0; i<outputSize; ++i)
            {
                auto label = i==y;
                auto aVal  = *(s+i);
                dc[d*outputSize+i] = aVal-label;
            }
        }
    }

};
//
class CECostFunc : public CostFunc
{
public:
    CECostFunc() {}
    
    string getName() const {return "CECFunc";}
    //
    float f(const vector<float>& a, LabelDataCItr dataStart, LabelDataCItr dataEnd) const
    {
        float val        = 0;
        auto  nbData     = distance(dataStart, dataEnd);
        auto  outputSize = a.size()/nbData;
        
        for (size_t d=0; d<nbData; ++d)
        {
            auto s = a.cbegin() + d*outputSize;
            auto y = (dataStart+d)->label;
            
            for (size_t i=0; i<outputSize; ++i)
            {
                auto label = i==y;
                auto aVal  = *(s+i);
                val+= -label*log(aVal) - (1-label)*log(1-aVal);
            }
        }
        return val;
    }
    //
    void df(const vector<float>& a, LabelDataCItr dataStart, LabelDataCItr dataEnd, vector<float>& dc) const
    {
        auto nbData     = distance(dataStart, dataEnd);
        auto outputSize = a.size()/nbData;
        
        vector<float> aloc(outputSize);
        vector<float> dcloc(outputSize);
        
        for (size_t d=0; d<nbData; ++d)
        {
            auto s = a.cbegin() + d*outputSize;
            auto y = (dataStart+d)->label;
            
            auto aLabel = *(s+y);
            for (size_t i=0; i<outputSize; ++i)
            {
                auto aVal  = *(s+i);
                dc[d*outputSize+i] = 1/(1-aVal);
            }
            dc[d*outputSize+y] = -1/aLabel;
        }
    }
};
//
class SMCostFunc : public CostFunc
{
public:
    SMCostFunc() {}
    
    string getName() const {return "SMCFunc";}
    //
    float f(const vector<float>& a, LabelDataCItr dataStart, LabelDataCItr dataEnd) const
    {
        float val        = 0;
        auto  nbData     = distance(dataStart, dataEnd);
        auto  outputSize = a.size()/nbData;

        for (size_t d=0; d<nbData; ++d)
        {
            auto s    = a.cbegin() + d*outputSize;
            auto e    = s + outputSize;
            auto y    = (dataStart+d)->label;
            auto maxA = *(max_element(s, e));
            
            vector<float> expA(outputSize);
            transform(s, e, expA.begin(), [maxA](auto x) {return exp(x-maxA);});
            auto sum = accumulate(expA.begin(), expA.end(), 0.);
            
            val -= log(expA[y]/sum);
        }
        return val;
    }
    //
    void df(const vector<float>& a, LabelDataCItr dataStart, LabelDataCItr dataEnd, vector<float>& dc) const
    {
        auto nbData     = distance(dataStart, dataEnd);
        auto outputSize = a.size()/nbData;

        for (size_t d=0; d<nbData; ++d)
        {
            auto s    = a.cbegin() + d*outputSize;
            auto e    = s + outputSize;
            auto y    = (dataStart+d)->label;
            auto maxA = *(max_element(s, e));
            
            vector<float> expA(outputSize);
            transform(s, e, expA.begin(), [maxA](auto x) {return exp(x-maxA);});
            auto sum = accumulate(expA.begin(), expA.end(), 0.);
            
            for (size_t o=0; o<outputSize; ++o)
                dc[d*outputSize+o] = expA[o]/sum;
            
            dc[d*outputSize+y] -=1;
        }
    }
};
//
class SVMCostFunc : public CostFunc
{
public:
    SVMCostFunc() {}
    
    string getName() const {return "SVMCFunc";}
    //
    float f(const vector<float>& a, LabelDataCItr dataStart, LabelDataCItr dataEnd) const
    {
        float val        = 0;
        auto  nbData     = distance(dataStart, dataEnd);
        auto  outputSize = a.size()/nbData;
        
        for (size_t d=0; d<nbData; ++d)
        {
            auto s = a.cbegin() + d*outputSize;
            auto y = (dataStart+d)->label;
            
            auto aLabel = *(s+y);
            for (size_t i=0; i<outputSize; ++i)
            {
                auto aVal  = *(s+i);
                val+= max(0.f,aVal-aLabel+1.f);
            }
            val -=1;
        }
        return val;
    }
    //
    void df(const vector<float>& a, LabelDataCItr dataStart, LabelDataCItr dataEnd, vector<float>& dc) const
    {
        auto nbData     = distance(dataStart, dataEnd);
        auto outputSize = a.size()/nbData;
        
        vector<float> aloc(outputSize);
        vector<float> dcloc(outputSize);
        
        for (size_t d=0; d<nbData; ++d)
        {
            auto s = a.cbegin() + d*outputSize;
            auto y = (dataStart+d)->label;
            
            auto aLabel = *(s+y);
            for (size_t i=0; i<outputSize; ++i)
            {
                auto aVal  = *(s+i);
                dc[d*outputSize+i] = aVal-aLabel>-1 ? 1. : 0.;
            }
            for (size_t i=0; i<outputSize; ++i)
            {
                auto aVal  = *(s+i);
                dc[d*outputSize+y] += aVal-aLabel>-1 ? -1. : 0.;
            }
        }
    }
};
//

}

#endif
