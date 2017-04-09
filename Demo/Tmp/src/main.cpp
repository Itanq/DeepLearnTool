
#include"hello.hpp"

int main()
{
    HI hello(std::string("Hello"));
    HI world(std::string("World"));

    hello.print(); 
    std::cout << " ";
    std::cout << " " ;
    world.print();
    std::cout << std::endl;
}
