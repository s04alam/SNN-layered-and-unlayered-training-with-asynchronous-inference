# SNN-layered-and-unlayered-training-with-asynchronous-inference
bachelor thesis (in progress)
warning: the file structure is a organized mess (for now), as I am currently working on eBrains, the most important file(s) are the jupyter files 

Two parts:

1. Full codebase from the published paper: https://openreview.net/challenge?redirect=%2Fforum%3Fid%3DmfmAVwtMIk. 

The requirement.txt file is incomplete, some minor changes in code (eg integer conversion), perhaps due to older python version. The newer requirements.txt file (thesis folder) should allow
to run the Github successfully.
All changes in code should have been commented as such. To replicate results, read the existing README file for further instructions.
This thesis used the existing Model 1 (layered) and Model 2 (unlayered) as reference, while reducing the input size from 700 to 128 and the size of the hidden layers from 128 to 64, to match the limitations for the BSS2 chip (number of synapses, fully connected layer, signed weights occupying two rows).
All design choices are tracked in a separate file, I can send it to you if you want (Mehrzad).

The trained models are stored as .pt (models/shd) weights are stored as .npz (z_thesis_files) and the inference results on F=128, F=8, F=1 (results/shd/29 or /30)

2. eBrains
I am still using the eBrains interface to access BSS2. However, I am limited to the Lab, thus jupyter files. There is a Github with the hxtorch library (link in the jupyter files).
In the jupyter files I import the model weights, I retain the structure of the software architecture (3x64-20) which works for inference, even on the full test set (of torch.no_grad.()).

With the weight scaling factor of approx 10, the neuron from software (snntorch) now matches the behaviour of the hxtorch neuron (mock=True), but the alignment between hxtorch mock=True and mock=False (hardware) is still in progress...
