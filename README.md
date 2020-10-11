# LT2316 H20 Assignment A1

Name: Mohamed Zarzoura

- [1. Notes on Part 1](#1-notes-on-part-1)
  - [1.1. Important notes](#11-important-notes)
  - [1.2. Tokenization](#12-tokenization)
  - [1.3. NE labeling](#13-ne-labeling)
  - [1.4. Checking the output](#14-checking-the-output)
  - [1.5. Issues with data](#15-issues-with-data)
    - [1.5.1. Introduction](#151-introduction)
    - [1.5.2. Plural Entity](#152-plural-entity)
    - [1.5.3. Typo in Sentence](#153-typo-in-sentence)
    - [1.5.4. charOffset Miss-reference](#154-charoffset-miss-reference)
    - [1.5.5. Summary of issues](#155-summary-of-issues)
- [2. Notes on Part 2](#2-notes-on-part-2)
  - [2.1. PPMI](#21-ppmi)
- [3. Notes on Part Bonus](#3-notes-on-part-bonus)

<br />

---

## 1. Notes on Part 1

### 1.1. Important notes

1. PLease run the code using the XML database located under my user folder `home/guszarzmo@GU.GU.SE/Corpora/DrugDrug-Interaction'. I have made several changes to the files to handle some issues. These issues are discussed in Section [1.5. Issues with data](#15-issues-with-data)

2. Due to the way that I choose to tokenize sentences, the method `<DataLoader>.get_random_sample()` does not provide the expected output. I wrote another method `<DataLoader>.get_random_sample_1()` that provides the same required function. You will find more details in section [1.4. Checking the output](#14-checking-the-output).

3. There are cases where two multiple-word NEs share the same word(s) in the sentence text. In two particular cases, each NE belongs to a different group. See section [1.3. NE labeling](#13-ne-labeling) for a detailed discussion.

4. All assignment's requirements are followed as far as I understood them, including the requirement of output dimensions. Regarding padding, to reduce the number of padding tokens used to equalize the sequences' length, the padding tokens will be added for each batch ---in the training, evaluation, and testing stage---. To comply with the assignment requirements and to be able to experiment with the idea mentioned above, I have added some methods and attributes as follows:

   1. `<DataLoader>.train`, `<DataLoader>.val` and ``<DataLoader>.test`:

      these `dict` hold lists of `token_ids`, `label-ids` for each token, and token's characters location `tuple`, in addition of a sequence length.

   2. `<DataLoader>.__get_sequence(tokens_df, labels_df)`:

      This method builds the `dicts` described above.

   3. `<DataLoader>.get_random_sample_1()`\
      `<DataLoader>.__combine_tokens(seq, loc)`\
      `<DataLoader>.__combine_labels(seq, loc)`\
      `<DataLoader>.__get_data_xml(myfile, sent_id)`

      These methods are used to check the parsing of data. See [1.4. Checking the output](#14-checking-the-output).

   4. `<DataLoader>.load_iter(self)`\
      `class DatasetObject(torch.utils.data.Dataset)`\
      `class DatasetField(DatasetObject)`

      Used to create torch dataset from `<DataLoader>.train`, `<DataLoader>.val` and `<DataLoader>.test`, discussed in item 1 above, using `torch.utils.data.Dataset`. I can batch and pad the sequences by using `torch.utils.data.DataLoader`.

   5. a flag `assignment_1=True`

      If activated, the `extract_features` function returns the features in the dimension required by the assignment. Otherwise, the features are without paddings along with PPMI embeddings for each token in dataset vocabulary. More details are in section [2. Notes on Part 2](#2-notes-on-part-2).

5. After the submission (during solving assignement-2), I realized the following issues:
   
   1. **PPMI calculation**: the values are calculated based on the splitted corpus, while the ppmi should be calculated based on the whole corpus.
   2. **`get_y` and `extract_features`**: `get_y` add pad tokens based on the max length in the split dataset itself. While `extract_features` add pads based on the max sequence length in the whole dataset
   
<br />

### 1.2. Tokenization

828 out-of 18491 entities have special characters within them, like hyphens, commas, brackets, etc.; spaces are not included in this counts. Example of these entities are:

1. `omega-agatoxin IVA`
2. `3-[(2-methyl-1,3-thiazol-4-yl) ethynyl] pyridine`
3. `N-[N-(3, 5-Difluorophenacetyl)-L-alanyl]-S-phenylglycine t-butyl ester`

Looking to example-3 above, we find a comma followed by space like what we would expect for punctuation. However, this comma is a part of the entity name. Generally speaking, it is hard to differentiate `hyphens` and punctuations that are part of a token/term from those used as stop words. Python libraries tokenizers were not able to make such differentiation. For example, `spacy` tokenize the entity is example-3 as follow:

```python
    [
        'N-[N-(3',      ',',        '5-Difluorophenacetyl)-L',  '-',
        'alanyl]-S',    '-',        'phenylglycine',            't',
        '-',            'butyl',    'ester'
    ]
```

In light of the above and considering that we are going to label each token, I decided to keep them in the tokenization process. Each special/ numeric character is a token for itself; i.e., the entity `5-FU` should be tokenized to be `[5, -, FU]`.

The method `<DataLoader>.__tokenizer_w_ents(text)` is responsible for tokenizing sentences as discussed by using the regular expression `r'(\W)`'. All tokens cases are lowered.

<br />

### 1.3. NE labeling

I used a BIO encoding to label the NEs. The method `_label_ner` is responsible for extracting NEs, labeling them, and returning them in a form that supports the assignment requirements.

There are several cases where two multiple-word NEs share the same word(s) in the sentence text. For example in file `Test/Test for DDI Extraction task/DrugBank/Methyclothiazide.xml`, sentence id: `DDI-DrugBank.d736.s8`, the sentence text is:

> Potentiation occurs with **ganglionic** or *peripheral adrenergic* ***blocking drugs***

The XML refer to two entities:

1. 'ganglionic adrenergic blocking drugs'
2. 'peripheral adrenergic blocking drugs'

Here, the two entities share the same tokens "blocking drugs" in ***bold italic*** in the sentence above. When such case is spotted by `__label_ner`, the label will be as follow:

```text

    with    **ganglionic**    or    peripheral    adrenergic    blocking drugs
     O        I-GROUP_N       O     I-GROUP_N     B-GROUP_N        GROUP_N

```

There are two special cases where the two NEs share the same words, but they belong to two different groups. As the BIO encoding cannot handle such cases, I decided to take them out. The reference to these two elements are:

1. File: `DrugDrug-Interaction/Train/DrugBank/Chlorothiazide_ddi.xml`
   Sentence id: DDI-DrugBank.d46.s19

2. File: `DrugDrug-Interaction/Train/DrugBank/Dopamine_ddi.xml`
   Sentence id: DDI-DrugBank.d325.s6

<br />

### 1.4. Checking the output

The provided method `<DataLoader>.get_random_sample()` does not provide the expected output. I am not sure why this happens. I speculate that the logic is based on assuming that the tokens are being split by spaces. In my case, a word ---which its boundary is two spaces--- may consist of multiple tokens with multiple labels. Thus it will be hard to print all of them in one line. So, I wrote another method `<DataLoader>.get_random_sample_1()` that combines the multiple tokens to form units that their boundary are a space, combine their labels by "|" and print all information in a table. Also, for reference, the data from the `XML` file is also printed.

A part of the output shown in `run.ipynb` is shown below. In the "tokens" table, in row 20, there are three tokens `[ amiloride, ), ,]` that have three labels. The parsing is then can be compared with information extracted from the `XML` file below.

```text
    --- Data parsed ---
        tokens            labels
    :   :                   :
    :   :                   :
    14  of                  O
    :   :                   :
    :   :                   :
    20  amiloride),         B-drug | O | O
    21  potassium           B-drug
    22  supplements,        O | O
    :   :                   :
    :   :                   :

    --- Data from xml file ---
    file path: /home/guszarzmo@GU.GU.SE/Corpora/DrugDrug-Interaction/Train/DrugBank/Losartan_ddi.xml

    As with other drugs that block angiotensin II or its effects, concomitant use of potassium-sparing diuretics (e.g., spironolactone, triamterene, amiloride), potassium supplements, or salt substitutes containing potassium may lead to increases in serum potassium.

              ner                    label
    0   potassium-sparing diuretics  group
    1   spironolactone               drug
    2   triamterene                  drug
    3   amiloride                    drug
    4   potassium                    drug
```

<br />

### 1.5. Issues with data

#### 1.5.1. Introduction

Several issues in the dataset has been found. To handel these issues, I either go around them using implementing logic in the code, fix the dataset it self, or take out the respective entry from the dataset. 

These issues are discussed in sections [1.5.2](#152-plural-entity) till [1.5.4](#154-charoffset-miss-reference). Changes made to the dataset files are summarized item by item in section [1.5.5. Summary of issues](#155-summary-of-issues).

#### 1.5.2. Plural Entity

The `<characters offsets>` in `xml` files discard the "*plur s*" in some of ne. For example in `Test/Test for DrugNER task/DrugBank/Tetracycline.xml'` the `<characters offsets>` of the ne **magnesium salicylates** is *100-119*. If the "plurals" is taken into account, the offset should be *100-**120***.

When such case is spotted during parsing, the "plural s s" is taken into account and the code will discard the xml `<characters offsets>`. This is to ensure that the text could be safely assembled after tokenization.

#### 1.5.3. Typo in Sentence

There are missing spaces in the `<sentence text>` field, mostly missing spaces. Such issues has been discovered as the extracted ne from the `<sentence text>` using `<characters offsets>` does not align with the `<entity text>`.

#### 1.5.4. charOffset Miss-reference

The `<characters offsets>` refer to either a part of a word or simply refers to wrong text. To align `<entity text>` field with the extracted text from the sentence, the `<characters offsets>` have been changed.

#### 1.5.5. Summary of issues

Files:

1. `Train/DrugBank/Clomipramine_ddi.xml`
2. `Train/DrugBank/Eszopiclone_ddi.xml`
3. `Train/DrugBank/Nevirapine_ddi.xml`
4. `Train/MedLine/11154900.xml`

    | File | sentence_id           | Issue                     | Description                                         | Action taken         |
    |------|-----------------------|---------------------------|-----------------------------------------------------|----------------------|
    | 1    | DDI-DrugBank.d238.s13 | Typo in Sentence          | missing space in sentence text                      | Change sentence text |
    | 1    | DDI-DrugBank.d238.s17 | Typo in Sentence          | missing space                                       | Change sentence text |
    | 1    | DDI-DrugBank.d238.s18 | Typo in Sentence          | missing space                                       | Change sentence text |
    |      |                       | charOffset Miss-reference | char offset of `e2` refer to wrong text in sentence | Change charOffset    |
    | 1    | DDI-DrugBank.d238.s20 | Typo in Sentence          | missing space                                       | Change sentence text |
    |      |                       | charOffset Miss-reference | char offset of `e1` refer to wrong text in sentence | Change charOffset    |
    | 1    | DDI-DrugBank.d238.s23 | Typo in Sentence          | missing space                                       | Change sentence text |
    |      |                       | charOffset Miss-reference | char offset of `e1` refer to wrong text in sentence | Change charOffset    |
    |      |                       | charOffset Miss-reference | char offset of `e2` refer to wrong text in sentence | Change charOffset    |
    | 2    | DDI-DrugBank.d216.s16 | charOffset Miss-reference | char offset of `e3` refer to wrong text in sentence | Change charOffset    |
    | 3    | DDI-DrugBank.d270.s30 | Typo in Sentence          | missing space                                       | Change sentence text |
    |      |                       | charOffset Miss-reference | char offset of `e1` refer to wrong text in sentence | Change charOffset    |
    | 4    | DDI-MedLine.d76.s9    | entity text               | the entity text is a part of token **CMC-Cys7.3**   | Change entity tex    |
    |      |                       | charOffset Miss-reference | char offset of `e1` refer to wrong text in sentence | Change charOffset    |

5. `Train/DrugBank/Dopamine_ddi.xml`

   **sentence_id**: DDI-DrugBank.d225.s7\
   **issue**: charOffset Miss-reference.

   The original "charOffset" refers to a part the adjective word "**dopaminergic**". The entity text `e3` refers to "**dopamine**", which is located at the end of the sentence. The character offset was changed.

    ```xml
    <sentence id="DDI-DrugBank.d325.s7" text="Butyrophenones (such as haloperidol) and phenothiazines can suppress the dopaminergic renal and mesenteric vasodilation induced with low dose dopamine infusion.">

            <entity id="DDI-DrugBank.d325.s7.e3" charOffset="73-80"
                type="drug" text="dopamine"/>
    ```

   <br />

6. `Train/DrugBank/Hydroflumethiazide_ddi.xml`

   **sentence_id**: DDI-DrugBank.d17.s6\
   **issue**: charOffset Miss-reference.

   The Original "charOffset" refers to a part of the adjective word "**preanesthetic**". The entity text `e2` refers should refer to "**anesthetic**" as the entity text would suggest. The character offset was changed.

   <br />

7. File: `Train/MedLine/11206417.xml`

   **sentence_id**: DDI-MedLine.d137.s2\
   **issue**: charOffset Miss-reference.

   The Original "charOffset" refers to a part of the word "**hyperinsulinaemia**". The entity text `e0` refers should refer to "**insulin**" which is located at the end of sentence. The character offset was changed.

   <br />

8. File: `Train/DrugBank/Liothyronine_ddi.xml`

   **sentence_id**: DDI-DrugBank.d54.s13\
   **issue**: charOffset Miss-reference.

   The Original "charOffset" refers to a part of the word "**thyroxine**" without the "e" at the end of the word. The entity "text" also refers to "thyroxin". As it is not feasible to refer to a part of token; each token has one label, I decided to take this element out.

   <br />

## 2. Notes on Part 2

Five features are selected; they are mainly linguistics features which I think they may have information :

1. POS tag of each token
2. Dependency tag of each token
3. The stem of each token. I tried to get the suffix and a prefix of each token but I did not find a library that can extract the prefix or multiple suffixes.
4. The head of the current token in the dependency graph.
5. The head's POS tag of the current token in the dependency graph.

### 2.1. PPMI

I expect that most of the NEs tokens may be `unknown` for a general pertained embeddings. Several tokens are just punctuation or a single character; also, these NEs are specialized expressions that are rare to be found in a general corpus. For this reason, I decided to include the co-occurrence count for tokens, as this will embed the context information for each token. A PPMI is then calculated for the token count.

I have investigated multiple reducing dimensions for the embeddings using the sum of variances explained by the reduced embeddings, and the results are summarized below. At 31% of the embeddings' original dimension, the principal components can explain more than 90% of the data variance.

| The reduced<br />Dimension | Reducing % | Sum of variances<br />explained |
| :---:                      | :---:      | :---:                           |
| 100                        |  1.02 %    | 16.36 %                         |
| 500                        |  5.12 %    | 44.50 %                         |
| 1000                       | 10.23 %    | 64.22 %                         |
| 3000                       | 30.70 %    | 93.05 %                         |
| 7000                       | 71.63 %    | 100.00 %                        |

The PPMI may be not considered as a feature rather that a way to represent a token in numerical way to be used by NN. Still not sure how to use them along with a pertained embeddings which are trained on more bigger corpus.

The PPMI calculation can be found in this [tutorial](https://github.com/henrywoo/MyML/blob/master/Copy_of_nlu_2.ipynb), the tutorial is licensed under the Apache License, Version 2.0.

## 3. Notes on Part Bonus

I used [venn](https://pypi.org/project/venn/) to draw the venn diagram. I installed venn by `pip install venn --user`
