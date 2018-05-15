
enum Platform{Windows, OSX, Linux};

#ifdef __CYGWIN__

const Platform platform = Linux;

#elif __GNU__

const Platform platform = Linux;

#elif  _WIN64

const Platform platform = Windows;

#elif __MACH__

const Platform platform = OSX;

#elif __APPLE__

const Platform platform = OSX;

#endif