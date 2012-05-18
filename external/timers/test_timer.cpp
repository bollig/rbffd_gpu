#include "timer_eb.h"
//#include <map> 
//#include <string> 

int main() 
{
    EB::TimerList tm;
    EB::Timer eb("timer noop (without map lookup)"); 
    EB::Timer eb2("Start Only (without map lookup)"); 
    EB::Timer eb3("Stop Only (without map lookup)"); 
    EB::Timer eb4("____ ignore (nomap timer started and stopped for tests) ____"); 

    tm["test"] = new EB::Timer("Time for all tests"); 
    tm["another"] = new EB::Timer("timer noop (includes map lookup)"); 
    tm["start"] = new EB::Timer("Start Only (includes map lookup)"); 
    tm["stop"] = new EB::Timer("Stop Only (includes map lookup)"); 
    tm["time"] = new EB::Timer("____ ignore (timer started and stopped for tests) ____"); 

    tm["test"]->start(); 

    tm["another"]->start(); 
    tm["another"]->stop(); 

    tm["start"]->start(); 
    tm["time"]->start();
    tm["start"]->stop(); 

    tm["stop"]->start(); 
    tm["time"]->start();
    tm["stop"]->stop(); 

    eb.start(); 
    eb.stop();

    eb2.start(); 
    eb4.start(); 
    eb2.stop(); 

    eb3.start(); 
    eb4.stop(); 
    eb3.stop(); 

    tm["eb"] = &eb; 
    tm["eb2"] = &eb2; 
    tm["eb3"] = &eb3; 
    tm["eb4"] = &eb4; 

    tm["test"]->stop(); 

    tm.writeAllToFile(); 
    tm.printAll(); 

    tm.writeToFile("alt_timer_log"); 

    return 0;
}
