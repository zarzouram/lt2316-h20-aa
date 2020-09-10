# NER Assignment

In this assignment you will working with the [Drug-Drug Interaction Corpus](https://core.ac.uk/download/pdf/82785218.pdf). The assignment will be to proprocess the data for Named Entity Recognition, i.e. to recognize pharmacological substances in text. This includes reading, parsing and structuring the data, data exploration and lastly creating features. 

Note that this dataset include data for two tasks, NER and NER-linking, but you will only be using it for NER. 

The dataset can be downloaded [here](https://canvas.gu.se/files/3359925/download?download_frd=1) (if this doesn work the data is located at canvas/files/LT2212 2020 materials/data/DDICorpus.zip).

Further information about the task can be found [here](https://www.aclweb.org/anthology/S13-2056.pdf) and [here](https://www.cs.york.ac.uk/semeval-2013/task9.html)

I suggest you start by forking the repo and then checking [run.ipynb](https://github.com/AxlAlm/ner_assignment/blob/master/run.ipynb). All work done in this assignment will be to make this file run as it is, except to minor changes which you are asked to make. This will give you a good idea of what is needed to be done.

### Part 1: Load dataset and explore a bit (8 Points)


In the dataset folder you will find xml files. All these are suppose to be parsed and structured. Fill in the functions in [DataLoader](https://github.com/AxlAlm/ner_assignment/blob/master/ass1/data_loading.py#L58) as described in the comments. For information how the data should be structured look at assertions in the [DataLoaderBase](https://github.com/AxlAlm/ner_assignment/blob/master/ass1/data_loading.py#L8)

Do not forget to document the choices you make with explanations why if needed.

For splitting the data use the premade Train/Test splits given and then take a part of the training as the val/dev set, as convention.


### part 2: Extract Features (8 points)

Now that we have the data we need to extract some features. Fill in [extract_features](https://github.com/AxlAlm/ner_assignment/blob/master/ass1/feature_extraction.py#L8) as instructed in the comments in the function.

In this part you are free to chose any type of feature you like, dont be afraid to be creative ;)! 

Document which feature(s) you use and why.


### Bonus Part: Extended Data Exploration (3 points)

As a bonus you are asked to extend the data exploration. As you might have notices some functions within [DataLoader](https://github.com/AxlAlm/ner_assignment/blob/master/ass1/data_loading.py#L58) are labled as beloning to the bonus part, fill these in so they plot whats requested in the comments. Uncomment the lines in the bonus part in [run.ipynb](https://github.com/AxlAlm/ner_assignment/blob/master/run.ipynb). 


If there are any issued with the current code, any bug or uncertainties, let me know either on Canvas, email or by making an issue here on github.

Best of luck 
Axel
