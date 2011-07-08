#ifndef __REALPATH_EXT_H__
#define __REALPATH_EXT_H__

#include <stdlib.h>
#include <string.h>
#include <limits.h>

// Extended realpath to support ~ at the beginning of the line
// From: http://www.dreamincode.net/forums/topic/218601-realpath-and-tilde/
char *realpathExt(const char *path, char *buff) {
    char *home;
    if (*path=='~' && (home = getenv("HOME"))) {
        char s[PATH_MAX];
        return realpath(strcat(strcpy(s, home), path+1), buff);
    } else {
        return realpath(path, buff);
    }
}

#endif 
