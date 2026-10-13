# S1 train / validation split

The TSV split is **not stored in this repo**. It lives with the dataset:

`../data/student_resource/dataset/train/` (workspace: `data/student_resource/dataset/train/`)

Read [`data/student_resource/dataset/split.md`](../data/student_resource/dataset/split.md).

`train_source1_train.tsv` / `train_source1_validation.tsv` (80/20 of train S1, seed 42, country × singleton vs matched). S2 and S3 are not split. Validation ground truth is for scoring only.
