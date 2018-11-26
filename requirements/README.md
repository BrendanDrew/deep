############
REQUIREMENTS
############

===========================================
Getting a useful TensorFlow wheel installed
===========================================

Unfortunately, TensorFlow seems to stay pretty far behind the times in terms of which versions
of CUDA, etc. they support for their own PIP releases. This means that unless you're equally
stuck in the past, training and inference is miserably slow. To work around this, I've tried to
do the following:

#. Provide a working docker file which can generate an Ubuntu 18.04 / CUDA 10 / cuDNN 7.3
TensorFlow wheel. If you are on Ubuntu, you can use this as a starting point for generating
your own wheel to install.

#. Provide this scaffolding to install a custom built TensorFlow as part of the Tox build.
This way, everything that runs inside the Tox-managed virtual environments gets the
benefit of a decent TensorFlow build.

For this to work, you will need to:

#. Do a build of TensorFlow that is appropriate to your platform. For Ubuntu 18.04 (and, with
minor modificals, also previous versions), you will need to

      #.  Install Docker (the community edition works great)
      #.  Run the docker file. Assuming you are already in `path/to/deep/dockerfiles`:

            ``
	    docker build -f BuildTensorFlowWheel.docker . -t tensor_flow_build && docker run --rm --mount type=bind,source=`pwd`,destination=/output tensor_flow_build
	    ``

      #.  Go get coffee. The TensorFlow build takes forever on a Core i7.

      #.  When the build is done, you will find a wheel file in the current working directory.
      	  Fix permissions, then copy into `path/to/deep/requirements`

      #.  Edit base.txt to list the specific wheel file you just copied into
      	  `path/to/deep/requirements`


#. Alternately, edit base.txt to just list tensorflow, or tensorflow_gpu if you don't want
   to build & run your own copy of TF.
      	  
