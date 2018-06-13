## Data

The dataset, saved model, filtered embeddings are available [here](https://drive.google.com/open?id=1zlJRe5GtOaX5RGbgGCL6wAgfHvDoPKNn).

## Execution:
Go to the `/src` directory and execute following commands. One should have the following files and directories `../data/embeddings` and `../data/definitions` present and `../saved/RNN_19.ckpt*`.
All of these are shared on the google drive link.

Command to Run RNN which will show results for RNN and RNN + SVD:

``` $ python train_definition_model.py --save_dir ../saved/RNN_19.ckpt --restore --evaluate ```


Command to Run SIF and SIF + SVD:

``` $ python arora.py ```

PS: This would easily be among the top five worst codes I have ever written.