set(PACKAGE_VERSION "1.4.2")
set(PACKAGE_VERSION_COMPATIBLE FALSE)
if(NOT "${PACKAGE_FIND_VERSION}" VERSION_LESS "${PACKAGE_VERSION}")
   if(NOT PACKAGE_FIND_VERSION_EXACT OR
         "${PACKAGE_FIND_VERSION}" VERSION_EQUAL "${PACKAGE_VERSION}")
      set(PACKAGE_VERSION_COMPATIBLE TRUE)
   endif()
   if(PACKAGE_FIND_VERSION_EXACT AND PACKAGE_VERSION_COMPATIBLE)
      set(PACKAGE_VERSION_EXACT TRUE)
   endif()
endif()
