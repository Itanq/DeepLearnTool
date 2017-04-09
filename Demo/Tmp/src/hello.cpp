
#include "hello.hpp"
#include <iostream>

HI::HI(std::string _str)
{
    s = _str;
}

void HI::print()
{
    std::cout << s;
}
