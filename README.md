# A Large, Gold Standard Dataset (LAGOS-AND) for Scholarly Author Name Disambiguation
This project presents a method to automatically generate a large-scale labeled dataset for author name disambiguation (AND) by leveraging authoritative resources ORCID and DOI. In addition, this project also provides ways to examine the last name variants in several large-scale bibliographic datasets/digital libraries, which a typical name synonym problem (can be understood that varied name mentions in citations point to same person, due to many reasons). Last, based on the large unbiased dataset, this project developed a disambiguation method, it can be used as a pre-trained model for many downstream tasks. 

### Dataset
Using the method, LAGOS-AND, a large, gold standard dataset for author name disambiguation has been built. The dataset is substantially different from existing ones. It contains 7.5M citations authored by 797K unique authors and shows close similarities to the entire Microsoft Academic Graph (213 million citations and 560 million authorship as of 2019) across six checks for gold standard validation, e.g., variation degree in the last name, ethnicity distribution, and domain distribution.

The dataset can be available from [here](https://zenodo.org/record/4568624), it is represented in two kinds of forms. The first one is full-name-block form, which arranges the dataset by ORCID iD and shared full name. The second is pairwise form, ambiguous authors are arranged in pair so that some classifier-based models can use it to capture the similarity between two authors.

#### Database Building
To build the dataset, the project considers several large-scale academic resources. It connects ORCID to the bibliographic datasets (i.e., Microsoft Academic Graph, Semantic Scholar and PubMed) by a credible paper identifier DOI. Then, based on the linkages, it can identify the author position of ORCID names in the linked bibliographic citations. Last, some fine-tunings are performed on it to ensure a high level of quality.

Note that the “dataset-generator.sql” provide the full implementation of the building procedures. It is a pure SQL implement that does not depend on any third-party packages; therefore, future versions of the dataset can be regenerated using the provided SQL scripts by updating the dependent data.

#### Our dataset v.s. Existing datasets
Creating a new dataset is painful. In AND researches, all existing datasets created by human annotators, and most datasets are either in limited scale or biased. However, our dataset has overcome these problems. It does not need human interventions in building the dataset. Moreover, by using the two comprehensive resources, the publishing history of a specific author (query DOIs by ORCID iD) and authors of a specific paper (query ORCID iDs by DOI) can be easily and credibly identified. Thus, with the large number of records in the credible resources, a large-scale dataset can be built. More importantly, the dataset considers more realistic aspects than existing datasets. It passed a series of rigorous gold standard validations, among which the two most important ones are synonym patterns and domains. The dataset contains a similar variation degree in last names and covers wide domain of research areas, as that represented in entire MAG. 

#### Dataset Structure
The block-based dataset contains the following fields:

| Field                        | Date Type     |
|------------------------------|---------------|
| block_fullname               | String        |
| author_group_orcid           | String        |
| author_group_idx_in_block    | Int           |
| citation_idx_in_author_group | Int           |
| doi                          | String        |
| pid                          | Int           |
| author_position              | Int           |
| author_name                  | String        |
| author_affiliation           | String        |
| coauthors                    | String Array  |
| coauthor_affliations         | String Array  |
| venue                        | String        |
| pub_year                     | Int           |
| paper_title                  | String        |
| paper_abstract               | String        |

"block_fullname" is taken from the credible full name (CFN) from the ORCID system, it is used to represent the block. Due to the fact that more than one authors can exist in a block, "author_group_orcid" is the ORCID iD of a specific author in a block, it is used to represent a group of citations (CG) that authored by this author, and "author_group_idx_in_block" denotes the order of CGs in a block. Similarly, "citation_idx_in_author_group" denotes the order of citation in a CG. "pid" is the paper ID in Microsoft Academic and Microsoft Academic Graph. "author_position" is identified by heuristics.

#### Last name variation
By leveraging authoritative resources (ORCID and DOI) and three literature databases (MAG, S2, PubMed), the last name variants can be credibly identified. Note MAG and S2 do not provide separate name components (first name, initial, last name), only full names can be obtained. Thus, to obtain the last name, this project utilized two high-quality, widely used name normalization tools [PHP-Name-Parser](https://github.com/joshfraser/PHP-Name-Parser) and [python-nameparser](https://github.com/derek73/python-nameparser).

### Pre-trained Model
The pre-trained disambiguation model can be accessible from [here](https://hub.docker.com/r/carmanzhang/lagos_and_model). Note that the model worth doing some experiments on general disambiguation tasks, because the large-scale training data contains wide domains of research areas, and considers more name variation patterns. To develop the model, the project uses a feature generation technique (i.e., a neural network) to model author similarity from citation content (typically, title and abstract). Then, the generated similarity is combined with other strong features to build the supervised disambiguation model using random forest.

#### Model Usage
- Make sure you have installed docker in your Linux machine. Then use the following commands to pull docker images from upstream.

```shell script
sudo docker pull carmanzhang/lagos_and_model:nn-submodel-1.0-alpha
sudo docker pull carmanzhang/lagos_and_model:1.0-alpha
```

- Or build images from sources

```shell script 
sudo docker build -t carmanzhang/lagos_and_model:nn-submodel-1.0-alpha -f ./nn-submodel.Dockerfile .
sudo docker build -t carmanzhang/lagos_and_model:1.0-alpha -f ./ml-submodel.Dockerfile .
```

- Then, you can run them locally

```shell script
sudo docker run --rm -it --net host carmanzhang/lagos_and_model:nn-submodel-1.0-alpha /bin/bash
sudo docker run --rm -it --net host carmanzhang/lagos_and_model:1.0-alpha bin/bash 
```

Note that, if successful, there will be two images, and two web services will be launched by them specifically. “lagos_and_model:nn-submodel-1.0-alpha” will listen at 38080 port, it is used to capture the similarity of citation content for pairwise author. Another web service “lagos_and_model:1.0-alpha” will listen at port 38081, this service is the final disambiguation service. Similar to “lagos_and_model:nn-submodel-1.0-alpha”, this service also output similarities [0-1] (0.5 is the decision value, score higher than o.5 is assumed to be same author) for pairwise citations, the score is predicted by the machine learning regressor (RF) inside the “lagos_and_model:1.0-alpha” container for describing how similar is the ambiguous authors. 

#### APIs and Usage
Both services provide HTTP API. Users can find and test them in the “deployment” folder. 
The APIs accept Json format messages and output Json format predictions. Note that the output predictions is in the same order as input. 
#### Content Similarity Predicting API
Input format
```json
{
   "contents":[
      [
         "content of ambiguous author X",
         "content of ambiguous author Y"
      ],

      ...

      [
         "content of ambiguous author M",
         "content of ambiguous author N"
      ]
   ]
}
```

Output format
```json
{
    "err_code":1,
    "err_msg":"",
    "scores":[
        0.23,

        …,

        0.9
    ]
}```



```json

```
#### Author Disambiguation API
Input format
```json
[
    [
        // the two names of ambiguous authors in the two citations   
        [
            "Amandine Descat",
            "Amandine Descat"
        ],
        // the publication year of the two citations
        [
            2017,
            2018
        ],
        // the full venue name of the two citations
        [
            "Journal of The International Society of Sports Nutrition",
            "PLOS ONE"
        ],
        // the affliations of the two citations
        [
            "Center of measurements and analysis (CMA), Faculty of Pharmaceutical Sciences, Université de Lille, Lille, France",
            ""
        ],
        // the content information (title+abstract) of the two citations
        [
            "Acute cocoa Flavanols intake has minimal effects on exercise-induced ...",
            "A pharmaco-metabolomics approach in a clinical trial of ALS ... "
        ]
    ]
	
	...,
	
]
```

Output format
```json
{
    "err_code":1,
    "err_msg":"",
    "scores":[
        0.83,

        …,

    ]
}
```

### Citation
If you used the dataset, method or model, please consider cite it.
```bibtex
@article{zhang2021lagos,
    title={LAGOS-AND: A Large, Gold Standard Dataset for Scholarly Author Name Disambiguation},
    author={Zhang, Li and Lu, Wei and Yang, Jinqing},
    journal={arXiv preprint arXiv:2104.01821},
    year={2021}
}
```
