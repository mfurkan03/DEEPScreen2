## DEEPScreen2: An Automated Tool for Drug-Target Interaction Prediction Using Deep Convolutional Neural Networks Fed by 2-D Images of Compounds

Accurately predicting drug-target interactions (DTI) possess critical importance in drug discovery and development, due to the labour-intensive and costly nature of conventional experimental screening techniques. A widely-utilised deep learning-based DTI prediction tool is DEEPScreen, which was previously developed by our group. DEEPScreen2, an advanced framework for DTI prediction (Figure 1), emerges as a DCNN solution for researchers regardless of technical background, building upon the groundwork laid by its predecessor, DEEPScreen.<br>
In this study, we proposed DEEPScreen2 (Figure 1) by offering numerous innovations over the previous implementation: (i) adopting 300x300 compounds images to increase the resolution and enable the capture of nuanced structural features; (ii) augmenting data by 36 different 10-degree rotations of the original compound images to render the model rotation invariant and thus more robust; (iii) fully automatizing the data preparation, data loading and inference/prediction procedures; (iv) updating the training, validation and test data via employing ChEMBL database v33 (as opposed to v23 in the old version); (v) introducing percentile-60 based data splitting as active and inactive data points with respect to the pChEMBL values, which extended the library over 800 target proteins.<br>
With its advanced image-based representations and flexible usability, DEEPScreen2 illustrates the relationship between artificial intelligence centric modeling and molecular understanding. We expect that DEEPScreen2 will be utilised in translational research, by scientists working on drug discovery and repurposing for fast and preliminary virtual screening of large chemical libraries. DEEPScreen2 is available as a programmatic tool together with its datasets and results at https://github.com/HUBioDataLab/DEEPScreen2.<br>

# How to use

## Prerequisites

- Python 3.x
- Install required packages:

  ```bash
  pip install requests pandas
  ```
## Usage
Run the script:
   
   ```bash
   python main_training.py --target_chembl_id "CHEMBL286" --assay_type "B" --pchembl_threshold 6.0 --output_file "activity_data.csv"
   ```

This will download and split the data and train the model using it.
