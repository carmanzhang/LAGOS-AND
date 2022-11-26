### Dataset

The dataset can be available from [here](https://zenodo.org/record/4568624), it is represented in two kinds of forms. The first one is full-name-block form, which arranges the dataset by ORCID iD and shared full name. The second is pairwise form, ambiguous authors are arranged in pair so that some classifier-based models can use it to capture the similarity between two authors.

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
