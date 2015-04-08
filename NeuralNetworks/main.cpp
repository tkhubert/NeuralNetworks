//
//  main.cpp
//  NeuralNetworks
//
//  Created by Thomas Hubert on 07/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//


#include "includes.h"

int main(int argc, const char * argv[])
{
    std::cout << "Hello, World!\n";
    
    SigmoidFunc sF;
    TanHFunc    tF;
    RLFunc      rF;
    
    std::cout << sF.f(1) << " " << tF.f(1) << " " << rF.f(1) <<std::endl;
    return 0;
}
