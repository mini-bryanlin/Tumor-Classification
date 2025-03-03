SHOW databases

-- @block
create table BC_DATA (
    id int
)
-- @block
LOAD DATA INFILE '~/Tumor-Classification/breast-cancer.csv' into table BC_DATA
FIELDS TERMINATED BY ','
IGNORE 1 LINES;

SELECT * FROM BC_DATA
-- @block
SHOW VARIABLES LIKE 'secure_file_priv';