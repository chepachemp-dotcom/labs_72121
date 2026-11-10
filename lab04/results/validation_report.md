# Validation report

Rows: 48842, columns: 15. Total violations: **0**. No rows were dropped.

| check | column | violations | detail |
|---|---|---|---|
| row_count | - | 0 | got 48842, expected 48842 |
| col_count | - | 0 | got 15, expected 15 |
| columns_present | - | 0 |  |
| type_int | age | 0 | ok |
| null_in_numeric | age | 0 |  |
| type_category | workclass | 0 | 8 distinct values |
| missing_info('?' or NaN; allowed) | workclass | 0 | 2799 missing (5.7%) |
| type_int | fnlwgt | 0 | ok |
| null_in_numeric | fnlwgt | 0 |  |
| type_category | education | 0 | 16 distinct values |
| missing_info('?' or NaN; allowed) | education | 0 | 0 missing (0.0%) |
| type_int | education-num | 0 | ok |
| null_in_numeric | education-num | 0 |  |
| type_category | marital-status | 0 | 7 distinct values |
| missing_info('?' or NaN; allowed) | marital-status | 0 | 0 missing (0.0%) |
| type_category | occupation | 0 | 14 distinct values |
| missing_info('?' or NaN; allowed) | occupation | 0 | 2809 missing (5.8%) |
| type_category | relationship | 0 | 6 distinct values |
| missing_info('?' or NaN; allowed) | relationship | 0 | 0 missing (0.0%) |
| type_category | race | 0 | 5 distinct values |
| missing_info('?' or NaN; allowed) | race | 0 | 0 missing (0.0%) |
| type_category | sex | 0 | 2 distinct values |
| missing_info('?' or NaN; allowed) | sex | 0 | 0 missing (0.0%) |
| type_int | capital-gain | 0 | ok |
| null_in_numeric | capital-gain | 0 |  |
| type_int | capital-loss | 0 | ok |
| null_in_numeric | capital-loss | 0 |  |
| type_int | hours-per-week | 0 | ok |
| null_in_numeric | hours-per-week | 0 |  |
| type_category | native-country | 0 | 41 distinct values |
| missing_info('?' or NaN; allowed) | native-country | 0 | 857 missing (1.8%) |
| type_category | income | 0 | 2 distinct values |
| missing_info('?' or NaN; allowed) | income | 0 | 0 missing (0.0%) |
| range | age | 0 | allowed [17, 90], observed [17, 90] |
| range | hours-per-week | 0 | allowed [1, 99], observed [1, 99] |
| range | capital-gain | 0 | allowed [0, 99999], observed [0, 99999] |
| range | capital-loss | 0 | allowed [0, 4356], observed [0, 4356] |
| allowed_labels | income | 0 | allowed ['<=50K', '>50K'] |
| exact_duplicate_rows (info, not dropped here) | - | 0 | 52 duplicates |