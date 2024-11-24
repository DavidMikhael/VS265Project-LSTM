# VS265Project-LSTM

This is the code for the first phase of my final project for the neural computation class. The dataset used here is from OpenNeuro. 

'Tommaso Fedele and Ece Boran and Valeri Chirkov and Peter Hilfiker and
Thomas Grunwald and Lennart Stieglitz and Hennric Jokeit and
Johannes Sarnthein (2020). Dataset of neurons and intracranial EEG from human
amygdala during aversive dynamic visual stimulation. OpenNeuro.
[Dataset] doi: 10.18112/openneuro.ds003374.v1.1.1'

It is an ECoG dataset collected from the amygdalae of nine subjects attending
a visual dynamic stimulation of emotional aversive vs neutral content. We are
interested in comparing fitting the data to an LSTM (to capture temporal
dependencies in the timeseries) and comparing the model's activations in
response to aversive vs neutral stimuli. Mainly, we are interesting in adding
a sparsity constraint and looking at the sparsity level for each stimulus type
