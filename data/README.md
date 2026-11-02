# Input data

The input CSV is not redistributed here. Its public download source and redistribution license have not been established from the available materials. Obtain permission to use a compatible copy before running the experiment; no public download link is claimed.

Save your local copy as `data/lectures_dataset.csv`, or supply its location with `--data`. The script does not download data. Processed fallback feature files are not substitutes for the raw 22-column input.

The recorded dataset has 11,548 rows and the following columns:

```text
auxiliary_rate
conjugate_rate
normalization_rate
tobe_verb_rate
preposition_rate
pronoun_rate
document_entropy
easiness
fraction_stopword_coverage
fraction_stopword_presence
subject_domain
freshness
title_word_count
word_count
most_covered_topic
topic_coverage
duration
lecture_type
has_parts
speaker_speed
silent_period_rate
median_engagement
```

`median_engagement` is the continuous target. The classification threshold is 0.2. `subject_domain`, `lecture_type`, and `has_parts` are categorical; missing values in `subject_domain` and `has_parts` become `unknown`. The topic URL is dropped. Other fields are numerical. Local data files are excluded by `.gitignore`.
