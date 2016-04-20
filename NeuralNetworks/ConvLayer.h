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
    
    // getters
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
    
    // main methods
    virtual void setFromPrev(const Layer* prevLayer) override;
    virtual void fwdProp    (const Layer* prevLayer) override;
    virtual void bwdProp    (      Layer* prevLayer) override;
    virtual void calcGrad   (const Layer* prevLayer) override;
    
protected:
    virtual void validate(const Layer* prevLayer) const;
    
    size_t width;
    size_t height;
    size_t depth;
    size_t prevDepth;
    size_t mapSize;
    size_t stride;
    
private:
    // naive
    void naiveFwdProp (const Layer* prevLayer);
    void naiveBwdProp (      Layer* prevLayer);
    void naiveCalcGrad(const Layer* prevLayer);
    
    // img2Mat
    void img2MatFwdProp (const Layer* prevLayer);
    void img2MatBwdProp (      Layer* prevLayer);
    void img2MatCalcGrad(const Layer* prevLayer);
    
    // img2Mat helper methods
    void genGradPrevAMat(size_t d, vec_r& prevAMat, const Layer* prevLayer) const;
};

}

#endif /* defined(__NeuralNetworks__ConvLayer__) */
