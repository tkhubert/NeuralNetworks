//
//  Trainer.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 11/05/2016.
//
//

#ifndef Trainer_h
#define Trainer_h

#include "NN.h"
#include "Optimizer.h"
#include "Regularizer.h"

namespace NN
{
    
class Trainer
{
public:
    // methods
    Trainer(const Optimizer& _optimizer, const Regularizer& _regularizer, size_t batchSize, size_t nbEpochs) :
        optimizer(_optimizer),
        regularizer(_regularizer),
        batchSize(batchSize),
        nbEpochs(nbEpochs)
    {};
    
    ~Trainer() {}
    unique_ptr<Trainer> clone() const { return make_unique<Trainer>(*this);}

    string getName() const
    {
        return optimizer.getName() + "_" + regularizer.getName() + "_" + getDetail();
    }
    string getDetail() const
    {
        return to_string(batchSize) + "_" + to_string(nbEpochs);
    }
    
    const auto& getOptimizer()   const {return optimizer;}
    const auto& getRegularizer() const {return regularizer;}
    auto        getBatchSize()   const {return batchSize;}
    auto        getNbEpochs()    const {return nbEpochs;}
    
private:
    const Optimizer&   optimizer;
    const Regularizer& regularizer;
    size_t batchSize;
    size_t nbEpochs;
};
    
}
#endif /* Trainer_h */
