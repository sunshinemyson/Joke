#include <iostream>
#include "CheckRuntime.hpp"

int _fake_main(int argc, char* argv[]) {

    zdl::DlSystem::Runtime_t rt = checkRuntime();
    
    std::cout<<"Fake_main"<<std::endl;
    return 0;
}
