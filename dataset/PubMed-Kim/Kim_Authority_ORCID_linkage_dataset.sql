create table if not exists and_ds.AUT_NIH
(
    Year             String,
    PMID             String,
    BylinePosition   String,
    MEDLINE_Name     String,
    NIH_ID           String,
    NIH_Name         String,
    Authority2009_ID String,
    Ethnea           String,
    Genni            String,
    AINI             String,
    FINI             String
) ENGINE = Log;

create table if not exists and_ds.AUT_ORC
(
    Year             String,
    PMID             String,
    BylinePosition   String,
    MEDLINE_Name     String,
    ORCID            String,
    ORCID_Name       String,
    Authority2009_ID String,
    Ethnea           String,
    Genni            String,
    AINI             String,
    FINI             String
) ENGINE = Log;

create table if not exists and_ds.AUT_SCT_info
(
    Year             String,
    PMID             String,
    BylinePosition   String,
    MEDLINE_Name     String,
    Authority2009_ID String,
    Ethnea           String,
    Genni            String,
    AINI             String,
    FINI             String
) ENGINE = Log;

create table if not exists and_ds.AUT_SCT_pairs
(
    PMID_1            String,
    Byline_Position_1 String,
    PMID_2            String,
    Byline_Position_2 String
) ENGINE = Log;

-- 312952 AUT_NIH.txt
-- 3076502 AUT_ORC.txt
-- 4732531 AUT_SCT_info.txt
-- 6214200 AUT_SCT_pairs.txt

-- 312951
-- 6214199
-- 3076501
-- 4732530
select count()
from and_ds.AUT_NIH
union all
select count()
from and_ds.AUT_ORC
union all
select count()
from and_ds.AUT_SCT_info
union all
select count()
from and_ds.AUT_SCT_pairs;

-- cat AUT_NIH.txt | dos2unix | clickhouse-client --password=root --input_format_allow_errors_ratio=0.01 --input_format_skip_unknown_fields=true --port=9001 --query='insert into and_ds.AUT_NIH FORMAT TSVWithNames'
-- cat AUT_ORC.txt | dos2unix | clickhouse-client --password=root --input_format_allow_errors_ratio=0.01 --input_format_skip_unknown_fields=true --port=9001 --query='insert into and_ds.AUT_ORC FORMAT TSVWithNames'
-- cat AUT_SCT_info.txt | dos2unix | clickhouse-client --password=root --input_format_allow_errors_ratio=0.01 --input_format_skip_unknown_fields=true --port=9001 --query='insert into and_ds.AUT_SCT_info FORMAT TSVWithNames'
-- cat AUT_SCT_pairs.txt | dos2unix | clickhouse-client --password=root --input_format_allow_errors_ratio=0.01 --input_format_skip_unknown_fields=true --port=9001 --query='insert into and_ds.AUT_SCT_pairs FORMAT TSVWithNames'

drop table if exists and_ds.AUT_NIH;
drop table if exists and_ds.AUT_ORC;
drop table if exists and_ds.AUT_SCT_info;
drop table if exists and_ds.AUT_SCT_pairs;

select *
from (
         select *
         from (select concat(PMID_1, '_', Byline_Position_1) as pm_ao1, concat(PMID_2, '_', Byline_Position_2) as pm_ao2
               from and_ds.AUT_SCT_pairs) any
                  inner join (select concat(PMID, '_', BylinePosition) as pm_ao1,
                                     MEDLINE_Name                      as MEDLINE_Name1,
                                     AINI                              as AINI1,
                                     FINI                              as FINI1
                              from and_ds.AUT_SCT_info) using pm_ao1
         ) any
         inner join (select concat(PMID, '_', BylinePosition) as pm_ao2,
                            MEDLINE_Name                      as MEDLINE_Name2,
                            AINI                              as AINI2,
                            FINI                              as FINI2
                     from and_ds.AUT_SCT_info) using pm_ao2;

-- 6214199	paired_authors
-- 1680310	num_citations
select count() as cnt, 'paired_authors' as name
from and_ds.AUT_SCT_pairs
union all
select arrayUniq(arrayConcat(groupArray(PMID_1), groupArray(PMID_2))) as cnt, 'num_citations' as name
from and_ds.AUT_SCT_pairs;

-- 3076501	num_citations
-- 268631	number_full_initial_based_blocks
-- 245754	num_author_groups
-- 197379	number_first_initial_based_blocks
select count() as cnt, 'num_citations' as name
from and_ds.AUT_ORC
union all
select count(distinct lowerUTF8(AINI)) as cnt, 'number_full_initial_based_blocks' as name
from and_ds.AUT_ORC
union all
select count(distinct lowerUTF8(FINI)) as cnt, 'number_first_initial_based_blocks' as name
from and_ds.AUT_ORC
union all
select count(distinct ORCID) as cnt, 'num_author_groups' as name
from and_ds.AUT_ORC
;

-- 312951	num_citations
-- 34206	num_author_groups
-- 37185	number_full_initial_based_blocks
-- 29243	number_first_initial_based_blocks
select count() as cnt, 'num_citations' as name
from and_ds.AUT_NIH
union all
select count(distinct lowerUTF8(AINI)) as cnt, 'number_full_initial_based_blocks' as name
from and_ds.AUT_NIH
union all
select count(distinct lowerUTF8(FINI)) as cnt, 'number_first_initial_based_blocks' as name
from and_ds.AUT_NIH
union all
select count(distinct NIH_ID) as cnt, 'num_author_groups' as name
from and_ds.AUT_NIH
;

-- name variations
-- 226588
select count()
from (
      select lowerUTF8(trimBoth(splitByString(',', MEDLINE_Name)[1])) as medline_lastname,
             lowerUTF8(trimBoth(splitByString('|', ORCID_Name)[1]))   as orcid_lastname,
             MEDLINE_Name,
             ORCID_Name
      from and_ds.AUT_ORC
      where medline_lastname != orcid_lastname)
union all
select count()
from and_ds.AUT_ORC
;


select count()
from (
      select lowerUTF8(trimBoth(splitByString(',', MEDLINE_Name)[1])) as medline_lastname,
             lowerUTF8(splitByChar('_', AINI)[1])                     as block_lastname
      from and_ds.AUT_ORC
      where medline_lastname != block_lastname)
;
select count()
from and_ds.AUT_ORC;

select arrayUniq(arrayConcat(groupArray(PMID1), groupArray(PMID2))) as cnt, 'num_citations' as name
from and.GS;