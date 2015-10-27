//
//  ConvLayer.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 15/08/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef __NeuralNetworks__ConvLayer__
#define __NeuralNetworks__ConvLayer__

#include "Layer.h"

namespace NN
{
    
class ConvLayer : public Layer
{
public:
    ConvLayer(size_t width, size_t height, size_t depth, size_t mapSize, size_t stride, const ActivationFunc& AFunc);
    virtual ~ConvLayer() {}
    
    string getName()      const override {return "ConvLayer";}
    string getDetails()   const override {return "";}
    LayerClass getClass() const override {return LayerClass::ConvLayer;}
    
    auto getWidth()   const {return width;}
    auto getHeight()  const {return height;}
    auto getDepth()   const {return depth;}
    auto getMapSize() const {return mapSize;}
    auto getStride()  const {return stride;}
    
    size_t getIdx (size_t d, size_t de, size_t h, size_t w)      const {return w+(h+(de+d*depth)*height)*width;}
    size_t getWIdx(size_t ode, size_t ide, size_t wh, size_t ww) const {return ww+mapSize*(wh+mapSize*(ide+prevDepth*ode));}
    
    virtual void setPrevLayer(Layer* layer) override;
    virtual void fwdProp()  override;
    virtual void bwdProp()  override;
    virtual void calcGrad() override;
    
protected:
    void validate() const;
    
    size_t width;
    size_t height;
    size_t depth;
    size_t prevDepth;
    size_t mapSize;
    size_t stride;
    
private:
    // naive
    void naiveFwdProp();
    void naiveBwdProp();
    void naiveCalcGrad();
    
    // img2Mat
    void img2MatFwdProp();
    void img2MatBwdProp();
    void img2MatCalcGrad();
    
    // img2Mat helper methods
    void genFwdPrevAMat(size_t d, vec_r& prevAMat) const;
    void genBwdDeltaMat(size_t d, vec_i& hIdxVec, vec_i& wIdxVec, vec_r& deltaMat) const;
    void genBwdWeightMat(vec_r& weightMat) const;
    void genBwdIdxVec(size_t pdim, size_t dim, vec_i& weightIdxVec) const;
    void genGradPrevAMat(size_t d, vec_r& prevAMat) const;
};

}

#endif /* defined(__NeuralNetworks__ConvLayer__) */
