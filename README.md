# XAI-Classifier-Forensics-Tool

Instructions:
1) Clone the repository

3) Download the pretrained models and sample images from [this link](https://drive.google.com/file/d/1XaUW1EtvuTfbOG38F1aRzWcpb1f4d0iO/view?usp=sharing) into `models` directory.

2) Create the conda environment with dependencies (if you don't have conda, you can also manually install the dependency in [environment.yml](https://github.com/Eric-mingjie/trojai-study-group-3/blob/main/environment.yml#L8))
```
conda env create -n trojai -f environment.yml
conda activate trojai
```

3) On the server, run the script `run.sh`:  
```
sbatch run.sh
```

4) Connect to the server via port 8080

5) Access `localhost:8080` in your web browser (e.g. chrome). Adjust the browser page size to get the best viewpoint.
