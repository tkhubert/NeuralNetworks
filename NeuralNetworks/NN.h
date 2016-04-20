//
//  includes.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef NeuralNetworks_includes_h
#define NeuralNetworks_includes_h

#include <iostream>
#include <fstream>
#include <sstream>

#include <cmath>
#include <cassert>
#include <ctime>

#include <algorithm>
#include <vector>
#include <string>
#include <random>
#include <memory>

namespace NN
{
    using namespace std;
    using real  = double;
    using vec_r = vector<real>;
    using vec_i = vector<int>;
    
    constexpr auto CHECKGRAD  = false;
    constexpr auto NBGRADTEST = 10;
    constexpr auto TWEAKSIZE  = 1e-2;
    constexpr auto GRADTOL    = 1e-6;
    constexpr auto TINY       = 1e-8;
    
    constexpr auto NAIVEFWD   = false;
    constexpr auto NAIVEBWD   = false;
    constexpr auto NAIVEGRAD  = false;
    
    constexpr auto GRADNEW    = false;
}

#endif
