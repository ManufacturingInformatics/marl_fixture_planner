calculateDeformationMARLTEST MATLAB Python Package

1. Prerequisites for Deployment 

Verify that MATLAB Runtime(R2023a) is installed.   
If not, you can run the MATLAB Runtime installer.
To find its location, enter
  
    >>mcrinstaller
      
at the MATLAB prompt.

Alternatively, download and install the Linux version of the MATLAB Runtime for R2023a 
from the following link on the MathWorks website:

    https://www.mathworks.com/products/compiler/mcr/index.html
   
For more information about the MATLAB Runtime and the MATLAB Runtime installer, see 
"Distribute Applications" in the MATLAB Compiler SDK documentation  
in the MathWorks Documentation Center.

Verify that a Linux version of Python 3.8, 3.9, and/or 3.10 is installed.

2. Installing the calculateDeformationMARLTEST Package

A. Change to the directory that contains the file setup.py and the subdirectory 
calculateDeformationMARLTEST. If you do not have write permissions, copy all its contents 
to a temporary location and change to that directory.

B. Execute the command:

    python setup.py install [options]
    
If you have full administrator privileges, and install to the default location, you do 
not need to specify any options. Otherwise, use --user to install to your home folder, or 
--prefix="installdir" to install to "installdir". In the latter case, add "installdir" to 
the PYTHONPATH environment variable. For details, refer to:

    https://docs.python.org/2/install/index.html

C. Set environment variables as follows:

In the following directions, replace MR/R2023a by the directory on the target machine where MATLAB is installed, or MR by the directory where the MATLAB Runtime is installed.

(1) Set the environment variable XAPPLRESDIR to this value:

MR/R2023a/X11/app-defaults


(2) If the environment variable LD_LIBRARY_PATH is undefined, set it to the following:

MR/R2023a/runtime/glnxa64:MR/R2023a/bin/glnxa64:MR/R2023a/sys/os/glnxa64:MR/R2023a/sys/opengl/lib/glnxa64

If it is defined, set it to the following:

${LD_LIBRARY_PATH}:MR/R2023a/runtime/glnxa64:MR/R2023a/bin/glnxa64:MR/R2023a/sys/os/glnxa64:MR/R2023a/sys/opengl/lib/glnxa64

3. Using the calculateDeformationMARLTEST Package

The calculateDeformationMARLTEST package is on your Python path. To import it into a 
Python script or session, execute:

    import calculateDeformationMARLTEST

If a namespace must be specified for the package, modify the import statement accordingly.
