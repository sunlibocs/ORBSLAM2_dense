# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/slbs/Desktop/code/ORB_SLAM2_dense

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/slbs/Desktop/code/ORB_SLAM2_dense/build

# Include any dependencies generated for this target.
include CMakeFiles/rgbd_my.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/rgbd_my.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/rgbd_my.dir/flags.make

CMakeFiles/rgbd_my.dir/Examples/RGB-D/rgbd_my.cc.o: CMakeFiles/rgbd_my.dir/flags.make
CMakeFiles/rgbd_my.dir/Examples/RGB-D/rgbd_my.cc.o: ../Examples/RGB-D/rgbd_my.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/slbs/Desktop/code/ORB_SLAM2_dense/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/rgbd_my.dir/Examples/RGB-D/rgbd_my.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rgbd_my.dir/Examples/RGB-D/rgbd_my.cc.o -c /home/slbs/Desktop/code/ORB_SLAM2_dense/Examples/RGB-D/rgbd_my.cc

CMakeFiles/rgbd_my.dir/Examples/RGB-D/rgbd_my.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rgbd_my.dir/Examples/RGB-D/rgbd_my.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/slbs/Desktop/code/ORB_SLAM2_dense/Examples/RGB-D/rgbd_my.cc > CMakeFiles/rgbd_my.dir/Examples/RGB-D/rgbd_my.cc.i

CMakeFiles/rgbd_my.dir/Examples/RGB-D/rgbd_my.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rgbd_my.dir/Examples/RGB-D/rgbd_my.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/slbs/Desktop/code/ORB_SLAM2_dense/Examples/RGB-D/rgbd_my.cc -o CMakeFiles/rgbd_my.dir/Examples/RGB-D/rgbd_my.cc.s

CMakeFiles/rgbd_my.dir/Examples/RGB-D/rgbd_my.cc.o.requires:

.PHONY : CMakeFiles/rgbd_my.dir/Examples/RGB-D/rgbd_my.cc.o.requires

CMakeFiles/rgbd_my.dir/Examples/RGB-D/rgbd_my.cc.o.provides: CMakeFiles/rgbd_my.dir/Examples/RGB-D/rgbd_my.cc.o.requires
	$(MAKE) -f CMakeFiles/rgbd_my.dir/build.make CMakeFiles/rgbd_my.dir/Examples/RGB-D/rgbd_my.cc.o.provides.build
.PHONY : CMakeFiles/rgbd_my.dir/Examples/RGB-D/rgbd_my.cc.o.provides

CMakeFiles/rgbd_my.dir/Examples/RGB-D/rgbd_my.cc.o.provides.build: CMakeFiles/rgbd_my.dir/Examples/RGB-D/rgbd_my.cc.o


# Object files for target rgbd_my
rgbd_my_OBJECTS = \
"CMakeFiles/rgbd_my.dir/Examples/RGB-D/rgbd_my.cc.o"

# External object files for target rgbd_my
rgbd_my_EXTERNAL_OBJECTS =

../Examples/RGB-D/rgbd_my: CMakeFiles/rgbd_my.dir/Examples/RGB-D/rgbd_my.cc.o
../Examples/RGB-D/rgbd_my: CMakeFiles/rgbd_my.dir/build.make
../Examples/RGB-D/rgbd_my: ../lib/libORB_SLAM2.so
../Examples/RGB-D/rgbd_my: /home/slbs/Desktop/software/Pangolin/build/src/libpangolin.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libGLEW.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libEGL.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libwayland-client.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libwayland-egl.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libwayland-cursor.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libdc1394.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libavcodec.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libavformat.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libavutil.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libswscale.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libavdevice.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libIlmImf.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/liblz4.so
../Examples/RGB-D/rgbd_my: ../Thirdparty/DBoW2/lib/libDBoW2.so
../Examples/RGB-D/rgbd_my: ../Thirdparty/g2o/lib/libg2o.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkDomainsChemistry-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneric-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersHyperTree-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallelFlowPaths-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersFlowPaths-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallelGeometry-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallelImaging-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallelMPI-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallelStatistics-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersProgrammable-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersPython-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/libvtkWrappingTools-6.3.a
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersReebGraph-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersSMP-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersSelection-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersVerdict-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkverdict-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkGUISupportQtOpenGL-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkGUISupportQtSQL-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkGUISupportQtWebkit-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkViewsQt-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOAMR-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOEnSight-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOExport-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkRenderingGL2PS-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkRenderingContextOpenGL-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOFFMPEG-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOMovie-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOGDAL-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOGeoJSON-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOImport-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOInfovis-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOMINC-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOMPIImage-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOMPIParallel-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOParallel-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIONetCDF-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOMySQL-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOODBC-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOParallelExodus-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOExodus-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkexoIIc-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOParallelLSDyna-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOLSDyna-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOParallelNetCDF-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOParallelXML-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOPostgreSQL-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOVPIC-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkVPIC-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOVideo-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOXdmf2-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkxdmf2-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkImagingMath-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkImagingMorphological-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkImagingStatistics-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkImagingStencil-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkInteractionImage-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkLocalExample-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkParallelMPI4Py-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkRenderingExternal-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeTypeFontConfig-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkRenderingImage-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkRenderingMatplotlib-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkWrappingPython27Core-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkPythonInterpreter-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkRenderingParallel-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallel-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkRenderingParallelLIC-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkParallelMPI-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkRenderingLIC-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkRenderingQt-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersTexture-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkGUISupportQt-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libQt5Widgets.so.5.9.5
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libQt5Gui.so.5.9.5
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libQt5Core.so.5.9.5
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolumeAMR-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersAMR-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkParallelCore-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolumeOpenGL-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libGLU.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libSM.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libICE.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libX11.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libXext.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libXt.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkTestingGenericBridge-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkTestingIOSQL-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOSQL-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkTestingRendering-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkViewsGeovis-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkViewsInfovis-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersImaging-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkRenderingLabel-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkGeovisCore-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOXML-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOXMLParser-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkInfovisLayout-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkInfovisBoostGraphAlgorithms-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkInfovisCore-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersHybrid-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkImagingGeneral-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkImagingHybrid-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOImage-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkDICOMParser-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkIOCore-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkmetaio-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkftgl-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libGL.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkImagingColor-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolume-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersStatistics-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkImagingFourier-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkalglib-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkCommonSystem-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtksys-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkWrappingJava-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-6.3.so.6.3.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libboost_thread.so
../Examples/RGB-D/rgbd_my: /usr/local/lib/libboost_date_time.so
../Examples/RGB-D/rgbd_my: /usr/local/lib/libboost_iostreams.so
../Examples/RGB-D/rgbd_my: /usr/local/lib/libboost_serialization.so
../Examples/RGB-D/rgbd_my: /usr/local/lib/libboost_chrono.so
../Examples/RGB-D/rgbd_my: /usr/local/lib/libboost_atomic.so
../Examples/RGB-D/rgbd_my: /usr/local/lib/libboost_regex.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpthread.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_common.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libexpat.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libgl2ps.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libjsoncpp.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_io.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_search.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_filters.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_features.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_ml.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libqhull.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_surface.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_registration.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_keypoints.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_tracking.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_recognition.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_stereo.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_apps.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_outofcore.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_people.so
../Examples/RGB-D/rgbd_my: /usr/local/lib/libboost_thread.so
../Examples/RGB-D/rgbd_my: /usr/local/lib/libboost_date_time.so
../Examples/RGB-D/rgbd_my: /usr/local/lib/libboost_iostreams.so
../Examples/RGB-D/rgbd_my: /usr/local/lib/libboost_serialization.so
../Examples/RGB-D/rgbd_my: /usr/local/lib/libboost_chrono.so
../Examples/RGB-D/rgbd_my: /usr/local/lib/libboost_atomic.so
../Examples/RGB-D/rgbd_my: /usr/local/lib/libboost_regex.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpthread.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_common.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libexpat.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libgl2ps.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libjsoncpp.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_io.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_search.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_filters.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_features.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_ml.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libqhull.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_surface.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_registration.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_keypoints.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_tracking.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_recognition.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_stereo.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_apps.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_outofcore.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpcl_people.so
../Examples/RGB-D/rgbd_my: /usr/lib/libOpenNI.so
../Examples/RGB-D/rgbd_my: /usr/lib/libOpenNI2.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libfreetype.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpython2.7.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libpng.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libjpeg.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libtiff.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libproj.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libnetcdf_c++.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libnetcdf.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libtheoraenc.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libtheoradec.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libogg.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libxml2.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/hdf5/openmpi/libhdf5.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libsz.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libz.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libdl.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/libm.so
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
../Examples/RGB-D/rgbd_my: librpg_open_remode.so
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_cudabgsegm.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_cudaobjdetect.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_cudastereo.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_shape.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_stitching.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_cudafeatures2d.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_superres.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_cudacodec.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_videostab.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_cudaoptflow.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_cudalegacy.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_calib3d.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_cudawarping.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_features2d.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_flann.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_objdetect.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_highgui.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_ml.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_photo.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_cudaimgproc.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_cudafilters.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_cudaarithm.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_video.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_videoio.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_imgcodecs.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_imgproc.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_core.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libopencv_cudev.so.3.2.0
../Examples/RGB-D/rgbd_my: /usr/local/lib/libboost_system.so
../Examples/RGB-D/rgbd_my: /usr/local/lib/libboost_filesystem.so
../Examples/RGB-D/rgbd_my: librpg_open_remode_cuda.so
../Examples/RGB-D/rgbd_my: /usr/local/cuda-9.0/lib64/libcudart_static.a
../Examples/RGB-D/rgbd_my: /usr/lib/x86_64-linux-gnu/librt.so
../Examples/RGB-D/rgbd_my: CMakeFiles/rgbd_my.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/slbs/Desktop/code/ORB_SLAM2_dense/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../Examples/RGB-D/rgbd_my"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rgbd_my.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/rgbd_my.dir/build: ../Examples/RGB-D/rgbd_my

.PHONY : CMakeFiles/rgbd_my.dir/build

CMakeFiles/rgbd_my.dir/requires: CMakeFiles/rgbd_my.dir/Examples/RGB-D/rgbd_my.cc.o.requires

.PHONY : CMakeFiles/rgbd_my.dir/requires

CMakeFiles/rgbd_my.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/rgbd_my.dir/cmake_clean.cmake
.PHONY : CMakeFiles/rgbd_my.dir/clean

CMakeFiles/rgbd_my.dir/depend:
	cd /home/slbs/Desktop/code/ORB_SLAM2_dense/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/slbs/Desktop/code/ORB_SLAM2_dense /home/slbs/Desktop/code/ORB_SLAM2_dense /home/slbs/Desktop/code/ORB_SLAM2_dense/build /home/slbs/Desktop/code/ORB_SLAM2_dense/build /home/slbs/Desktop/code/ORB_SLAM2_dense/build/CMakeFiles/rgbd_my.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/rgbd_my.dir/depend

