### PINN: Associative Neural Networks based on Pseudo-inverse (aka Projection) Learning Rules

See http://www.videorecognition.com/memory/pinn/ for more details

**What's in name? What it is ?**

These neural networks are referred to as:

-pseudo-inverse networks - for using Moore-Penrose pseudo-inverse V+ in computing   the synapses   

-projection networks - for synaptic (weight) matrix C=VV+  being   the projection   

- Hopfield-like networks - for being binary and fully-connected in the stage of  learning 

- recurrent networks - for evolving in time, based on external input and internal  memory  attractor-based networks - for storing patterns as attractors (i.e. stable states of      the network)       

- dynamic systems - for allowing the dynamic systems theory to be applied   

- associative memory - for being able to memorize and recall and patterns      associatively,      just as much as humans do     

- PINN memory - pseudo-inverse neural network - based associative memory.  

**Eight reasons to like pseudo-inverse neural networks (PINN)**

It evolved from the network of  formal neurons as defined by Hebb in 1949 and has many     
parallels with biological memory mechanisms.    

It provides an   analytical (close-form) solution, the approximation of which is a well-known delta learning rule.   

The performance of the network can be very nicely   analytically examined and  improved.  

It can   update the memory on fly  in real time, storing patterns with a non-iterative learning rule.    

For a network of size N, it allows one to  store (with error-correction!)   up to    for  M=70%N prototypes,   using the desaturation mechanism.   

At the same time when  dealing with a continuous stream of data, thanks to the desaturation mechanism,  the network will never get saturated and will always  maintain the 20%N capacity, always being able to converge to 20%N main attractors.  

All this makes the network very  suitable for  on-line memorization and  recognition, and also for   preprocessing binary patterns, as noise removing filter.   

Starting from a fully connected configuration, it can be converted to a Small-World     configuration of the much smaller size.  

Critical example:  Applying PINN memory for  Perceptual Vision   Systems and  Face Recognition in Video. 

2. Open Source codes and library:  

>> Basic code for associative storing and retrieving patterns with fully-connected PINN memory:      pinn.cpp 
This is a one-file code, which is entirely self-contained and ready to be compiled and run.      

What it does:       
It defines the CPINN class, which is used to create an associative memory  
It takes a binary pattern, one at a time, and adds it to the memory - very fast (no iterating!)  
It computes the level of remaining memory capacity, which is used to evaluate the network's recognition capability and to allow one to reconfigure the network, if required, to ensure the best performance (analytically!)  
It recognizes stored patterns: when presented a new pattern, the memory finds the pattern the closest to it in associative sense and tells whether the retrieved pattern has been previously seen or not. 

>>Associative Neural Networks Library:      library.html   

This is a C++ implementation of  most known high-performance associative neural networks models, including those based on sparse     architectures and pseudo-inverse learning. 

For more details see:     

Dmitry Gorodnichy, Investigation and Design of High Performance Fully-Connected Neural Networks, Candidate of Phys.&Math. Science (Ph.D.) Dissertation, Online: https://sites.google.com/site/dmitrygorodnichy/projects/PhD-PINN

High Performance Associative Neural Networks: Overview and Library. Alexey Dekhtyarenko and Dmitry Gorodnichy, Canadian Conference on Artificial Intelligence (AI'06), Quebec   City, Canada, July 7-9, 2006   
