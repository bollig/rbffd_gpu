#ifndef _TIME_EB_H_
#define _TIME_EB_H_

// gettimeofday: measured in sec/microsec: wall clock time
// irrespective of CPU/system/threads, etc.


#ifdef WIN32
#include <time.h>
#include <Windows.h>
#include "gtod_windows.h"
#else
#if (__PGI)
#include <time.h>
#else 
#include <sys/time.h>
#endif
#endif

#ifdef WIN32
#if defined(rtps_EXPORTS)
#define RTPS_EXPORT __declspec(dllexport)
#else
#define RTPS_EXPORT __declspec(dllimport)
#endif 
#else
#define RTPS_EXPORT
#endif

#include <string>
#include <vector>
#include <stdio.h>
#include <map> 
#include <string> 

#if (__PGI)
#define CLOCK_T_TYPE __clock_t
#else 
#define CLOCK_T_TYPE clock_t
#endif 

namespace EB {

    class Timer
    {
        public:
            //static std::vector<Timer*> timeList;

        private:
            struct timeval t_start, t_end;
            double elapsed;
            float t;
            CLOCK_T_TYPE t1;
            CLOCK_T_TYPE t2;
            float scale;
            std::string name;
            std::string unit;
            int count;
            int nbCalls;
            int offset;

        public:
            // nbCalls: how many calls before resetting the clock
            // if nbCalls not -1, print time after nbCalls calls
            // offset: how many calls to ignore
            Timer();
            Timer(const char* name, int offset=0, int nbCalls=-1);
            Timer(const char* name, float initial_time, int offset=0, int nbCalls=-1);
            Timer(const Timer&);
            ~Timer();
            void reset();
            void begin();
            void end();
            int getCount() { return count;}

            void stop() { end(); }
            void start() { begin(); }

            void set(float t); //add a time from an external timer (GPU)

            //static void printAll(FILE* fd=stdout, int label_width=40);
            void print(FILE* fd=stdout, int label_width=40);
            //void writeAllToFile(std::string filename="timer_log"); 
            void printReset();
            std::string& getName() { return name; } 
    };



#if 0
    class TimerList : public std::map<std::string, EB::Timer*>
    {
        public: 
            virtual ~TimerList() { 
                if (!this->empty()) {
                    this->printAll(); 
                    this->clear();
                }
            }

            bool contains (std::string timer_name) { 
                std::map<std::string, EB::Timer*>::iterator iter = this->find(timer_name);
                if ( this->end() != iter ) {
                    return true; 
                } 
                return false; 
            }

            void writeAllToFile(std::string filename="timer_log") 
            {
                // Get the max label width so we can show all columns the same
                // width and show the FULL label for each timer
                unsigned int label_width = 50; 
                for (TimerList::iterator i=this->begin(); i != this->end(); i++) {
                    if (i->second->getName().length() > label_width) {
                        label_width = i->second->getName().length(); 
                    }
                }
                FILE* fd = fopen(filename.c_str(), "w"); 
                this->printAll(fd, label_width); 
                fclose(fd); 
            }

            void writeToFile(std::string filename="timer_log") { 
                this->writeAllToFile(filename);
            }
#if 0
            void writeAllToFile(std::string filename="timer_log") {
                (*(this->begin())).second->writeAllToFile(filename); 
            } 
            void printAll() {
                (*(this->begin())).second->printAll(); 
            } 
#endif 
            void printAllNonStatic(FILE* fd=stdout, int label_width=40)
            { this->printAll(fd, label_width); }
            void printAll(FILE* fd=stdout, int label_width=40)
            {
                if (fd != stdout ) {
                    fprintf(fd, "====================================\n"); 
                    fprintf(fd, "Timers [All times in ms (1/1000 s)]: \n"); 		
                    fprintf(fd, "====================================\n\n");     
                }
                for (TimerList::iterator i=this->begin(); i != this->end(); i++) {
                    Timer& tim = *(i->second); 
                    tim.print(fd, label_width);
                }

                if (fd != stdout ) {
                    fprintf(fd, "\nNOTE: only timers that have called Timer::start() are shown. \n");
                    fprintf(fd, "      [A time of 0.0 may indicate the timer was not stopped.]\n"); 
                    fprintf(fd, "====================================\n"); 
                }
            }
            virtual void clear() {
                for (TimerList::iterator i=this->begin(); i != this->end(); i++) {
                    delete(i->second);
                }
                std::map<std::string, EB::Timer*>::clear();
            }
    };
#endif 

};
#endif
