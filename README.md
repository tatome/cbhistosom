# cbhistosom
A cyclic-topology batch-update SOM-like ANN algorithm that uses histograms to learn the statistics of its input.

This ANN algorithm was developed for my [PHD dissertation](http://ediss.sub.uni-hamburg.de/volltexte/2015/7645/pdf/Dissertation.pdf) "One Computer Scientist's (Deep) Superior Colliculus."

It was used to implement adaptive binaural sound-source localization on a robot.  Unfortunately, I can't publish the sound preprocessing code (yet), and publishing the training code without the preprocessing code does not make a lot of sense, so this is _just_ the code for the network itself.

Refer to the [relevant publications](http://www.tatome.de/index.php?id=academic), in particular Chapters 9 and 10 of my dissertation for how to use it.
