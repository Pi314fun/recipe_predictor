# Recipe Predictor

An application that processes a dataset of Indian food recipes, trains machine learning models on the data, and provides a GUI for users to predict recipes based on input ingredients.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Recipe Predictor is a comprehensive solution that spans from raw data preprocessing to an interactive GUI application. It preprocesses an Indian food recipes dataset, establishes a machine learning pipeline for training multiple models, and wraps everything into a user-friendly application for recipe predictions.
[Kaggle](https://www.kaggle.com/datasets/kanishk307/6000-indian-food-recipes-dataset)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Pi314fun/recipe_predictor.git
   ```

2. Navigate to the directory:

   ```bash
   cd recipe_predictor
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. First, preprocess the dataset:

   ```bash
   python preprocessing.py
   ```

2. Next, run the machine learning pipeline script:

   ```bash
   python run_project.py
   ```

3. Finally, launch the GUI application:

   ```bash
   python gui_application.py
   ```

Follow the on-screen instructions in the GUI to input ingredients and get recipe predictions.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
