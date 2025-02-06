# README

This is an explanation file of how to train and test CT(Compositional Transfer) for source free domain adaptation tasks in 

**<u>Learning Compositional Transferability of Time Series for Source-Free Domain Adaptation</u>**.

## Document structure

```python
root
├── checkpoints
   	  ├── [X]to[X]_SFDA_[dataset name]_[stageX]_Exp_0
   	  └── ...
├── data_provider
	  ├── data_factory.py
      └── data_loader.py
├── datasets
	  ├── [dataset name]
    				├── train[X].pt
        			├── test[X].pt
        			└── ...
      └── ...
├── exp
	  ├── trainer_basic.py
      ├── trainer_stage1.py
      ├── trainer_stage2.py
      ├── trainer_stage3.py
      └── trainer_tta.py
├── results
	  ├── [X]to[X]_SFDA_[dataset name]_[stageX]_Exp_0
├── run
	  ├── [dataset name]
         ├── train
                ├── [X]to[X].sh
                ├── ...
                └── total.sh
         └── tta
                ├── [X]to[X]tta.sh
                └── ...
      └── ...
├── utils
	  ├── compute.py
      ├── losses.py
      ├── metrics.py
      └── ...
├── model.py
├── modules.py
├── run.py
└── README.md
```

The structure of our file is below. Here is the explaination:

- The model and the executable files are saved in the top-level directory, named **<u>model.py</u>** and **<u>run.py</u>**, respectively.
- The **<u>datasets</u>** folder contains subfolders for the datasets, with each dataset folder storing the corresponding **.pt** format time series data, named in the format train/test[X], where X is the domain number.
- The **<u>data_provider</u>** folder contains the code for loading the datasets and performing the corresponding preprocessing.
- The <u>**exp**</u> folder contains the code for three training stages and one TTA stage, with the filenames **<u>trainer_stage[X].py</u>** and **<u>trainer_tta.py</u>**, respectively.
- The **<u>run</u>** folder contains two subfolders, **<u>train</u>** and **<u>tta</u>**, which store the corresponding cross-domain training and TTA execution commands for each dataset. The command format is **<u>[X]to[X].sh</u>**. Additionally, we have placed a **<u>total.sh</u>** file in each dataset folder within the trainer, which can be run to complete all three stages of cross-domain training for that dataset at once.
- The **<u>utils</u>** folder contains certain function codes needed throughout the entire operation process. For example, **<u>compte.py</u>** includes some calculation functions, while <u>**metrics.py**</u> stores functions for calculating metrics.
- After running, the model will be stored in the **<u>checkpoints</u>** folder, while the results of the run will be output to the <u>**results**</u> folder. The storage format in both folders is [**X]to[X]\_SFDA\_[dataset name]_[stageX]\_Exp_0**, where the first two Xs represent the domain numbers, and the last X represents the stage number of the training.

## Requirmenets

Our model training and testing need to be conducted in the following environment:

- Python3
- Pytorch==2.0
- Numpy==1.26.4
- scikit-learn==1.4.2
- Pandas==2.2.2
- torchmetrics==1.4.0.post0
- scipy==1.13.0

## Datasets

### Available Datasets

We used four public datasets in this study. We also provide the **preprocessed** versions as follows:

- [SSC](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/UD1IM9)
- [UCIHAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/0SYHTZ)
- [MFD](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/PU85XN)

## Training procedure

If you want to train an CT model, you need to first place the dataset in the corresponding folder (according to the requirements of the file structure mentioned earlier), and then run the run.py file using the python command while passing the parameters you want to set. Below is an example of running the training.

```bash
python -u run.py \
  --task_name classification \
  --is_training 1 \				# train: 1; test: 0
  --root_path ./FD/ \
  --model_id 0to1 \				# domain 0 -> 1
  --model SFDA \
  --data FDA \					# MFD:FDA; SSC:SSC; HAR:HAR
  --batch_size 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.005 \
  --num_classes 3\
  --win_size 5120\				# series length
  --win_step 5120\				# series length
  --train_domain train_0.pt\	# train the model on domain 0
  --test_domain test_0.pt\		# test the model on domain 0
  --stage 'stage1'\				# stage1/stage2/stage3/tta
  --train_epochs 8
```

To facilitate training, we have stored all datasets and the three-phase training for cross-domain scenarios in the **<u>run</u>** folder. You only need to select the corresponding bash file to run in order to complete the respective cross-domain training. After that, the model will be saved in the <u>**checkpoints**</u> folder.



## Testing procedure

If you want to test your trained cross-domain model, you can also follow the running code mentioned above. Just change the is_training parameter to 0, set test_domain to the name of the domain data file you want to test (ensuring it is stored under the corresponding dataset), and set the stage parameter to 'stage3'. Running this will load trained model and yield the corresponding test results. The example command is as follows:

*Reminding: you should train an adaptation model before testing.*

```bash
python -u run.py \
  --task_name classification \
  --is_training 1 \				# train: 1; test: 0
  --root_path ./FD/ \
  --model_id 0to1 \				# domain 0 -> 1
  --model SFDA \
  --data FDA \					# MFD:FDA; SSC:SSC; HAR:HAR
  --batch_size 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.005 \
  --num_classes 3\
  --win_size 5120\				# series length
  --win_step 5120\				# series length
  --train_domain train_1.pt\
  --test_domain test_1.pt\		# test the model on domain 1
  --stage 'stage3'\				
  --train_epochs 8
```

We have also included a test command in each training bash file, allowing you to quickly obtain the test results immediately after training.

In addition, if you want to obtain the results after instance-wise adaptation, you need to set the stage parameter in the test command to 'tta', and then add the following three parameters:

```bash
python -u run.py \
  --task_name classification \
  --is_training 0 \
  # ... These parameters are the same as those in the test bash command.
  --test_domain test_1.pt\
  --stage 'tta'\
  --train_epochs 8 \
  --patience 20 \
  --tta 1 \					# using tta: 1; not using tta: 0
  --delta 0.001 \			# ratio of each step, 0.001 denotes 0.1%
  --N 10					# number of step
```

The corresponding commands are stored in the tta folder under the <u>**run**</u> directory.





