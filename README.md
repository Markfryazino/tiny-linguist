# The Evolution of Syntactic Knowledge in Language Models

This project explores trains a small language model on the TinyStories dataset and explores the dynamics of its performance on the tasks involving syntactic knowledge.

[Link to the paper](https://drive.google.com/file/d/1SNm0_BeSs8Fn2Cj2fAH-0IhrorpRjqcx/view?usp=sharing)

### How to reproduce the experiments

1. Install the packages:
    ```
    pip install -r requirements.txt
    ```

2. Export the data path:
    ```
    export DATA_PATH=your_path
    ```

3. Download the data from HuggingFace website:
    ```
    wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
    wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
    ```

    Store them in $DATA_PATH/data.

4. Train the model:
    ```
    python train.py
    ```

5. Sample the validation data: run the notebook `handle_val_data.ipynb`. You will need to replace some paths in this notebook (and all the others) with yours.

6. Label the validation data:

    ```
    python label.py
    ```

7. Run the model checkpoints on the tasks:
    ```
    python extract_representations.py --data_path your_path --model_path your_path
    ```

8. Compute the quality of the model predictions:
    ```
    python compute_quality.py
    ```

9. Generate texts for the discriminator:
    ```
    python generate_texts.py --data_path your_path --model_path your_path
    ```

10. Train the discriminator: run the notebook `disriminator.ipynb`.

11. Build the plots: run the notebook `results.ipynb`.