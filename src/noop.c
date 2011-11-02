// NO OP empty file. Allows us to call ADD_LIBRARY() in CMake on OSX with no files to compile
// Will give a warning that this file contains no symbols, so we declare a variable here to make
// a global variable
int __noop_noop_dummy_var; 
