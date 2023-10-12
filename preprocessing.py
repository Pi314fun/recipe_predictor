import pandas as pd
import re


def clean_ingredients(ingredient_str):
    """
    Clean ingredient strings by removing descriptors, numbers, and unnecessary characters.

    Args:
    - ingredient_str (str): Raw ingredient string to clean.

    Returns:
    - str: Comma-separated cleaned ingredients.
    """
    # Split the input ingredient string by commas
    ingredients = ingredient_str.split(',')

    # List of descriptors to remove from ingredient list
    descriptors = [
        'teaspoon', 'tablespoon', 'cup', 'grams', 'pieces', 'sliced', 'chopped',
        'finely', 'diced', 'desi', 'gms', 'tbsp', 'tsp', 'ml', 'inch', 'large',
        'medium', 'small', 'shredded', 'to taste', 'roughly', 'fresh', 'peeled',
        'de seeded', 'deseeded', 'crushed', 'whole', 'cubes', 'cube', 'round',
        'grated', 'powder', 'optional', 'dry', 'washed', 'soaked', 'cooked',
        'uncooked', 'ripe', 'unripe', 'frozen', 'thin', 'thick', 'cleaned', 'thinly',
        'for', 'into', 'and', 'as per taste', 'to', 'cut', 'overnight', 'leaves',
        'into strips', 'ti'
    ]

    cleaned_ingredients = []
    for ingredient in ingredients:
        # Remove digits
        ingredient = re.sub(r'\d+', '', ingredient)
        # Replace dashes and slashes with spaces
        ingredient = re.sub(r'[-/]', ' ', ingredient)
        # Remove descriptor words
        ingredient = ' '.join(
            [word for word in ingredient.split() if word not in descriptors]
        )
        cleaned_ingredients.append(ingredient.strip())

    return ', '.join(cleaned_ingredients)


def clean_recipe(recipe_str):
    """
    Clean recipe name string.

    Args:
    - recipe_str (str): Raw recipe name string to clean.

    Returns:
    - str: Cleaned recipe name.
    """
    # Remove content inside parentheses
    cleaned = re.sub(r'\(.*?\)', '', recipe_str)
    # Extract the string before " - " pattern
    cleaned = cleaned.split(" - ")[0]
    return cleaned.strip()


def preprocess_data(filepath, output_filepath):
    """
    Preprocess the dataset by cleaning and filtering necessary columns and rows.

    Args:
    - filepath (str): Path to the input Excel dataset.
    - output_filepath (str): Path to save the processed CSV dataset.

    Returns:
    - None: Saves the processed data to the specified output path.
    """
    # Load dataset
    data = pd.read_excel(filepath)

    # Fill NA values in 'TranslatedIngredients' column
    data['TranslatedIngredients'].fillna("", inplace=True)

    # List of meats to filter out
    meats = ["chicken", "mutton", "lamb", "beef", "pork", "fish",
             "prawn", "shrimp", "goat", "crab", "duck", "turkey"]

    # Filter out non-vegetarian rows based on ingredients and diet
    filtered_data = data[~data['TranslatedIngredients'].str.lower(
    ).str.contains('|'.join(meats))]
    filtered_data = filtered_data[~filtered_data['Diet'].str.contains(
        "Non Vegeterian|Non Vegetarian", case=False)]

    # Drop unnecessary columns and rows
    filtered_data.drop(columns=['TranslatedInstructions'], inplace=True)
    filtered_data.dropna(inplace=True)

    # Filter out rows containing Devanagari script characters
    filtered_data = filtered_data[~filtered_data["TranslatedIngredients"].str.contains(
        r'[ऀ-ॿ]')]

    # Clean ingredient and recipe columns
    filtered_data["TranslatedIngredients"] = filtered_data["TranslatedIngredients"].apply(
        clean_ingredients)
    filtered_data["TranslatedRecipeName"] = filtered_data["TranslatedRecipeName"].apply(
        clean_recipe)

    # Rename columns for clarity
    column_rename_map = {
        "TranslatedRecipeName": "Recipe",
        "TranslatedIngredients": "Ingredients",
        "PrepTimeInMins": "PrepTime",
        "CookTimeInMins": "CookTime",
        "TotalTimeInMins": "TotalTime"
    }
    filtered_data.rename(columns=column_rename_map, inplace=True)

    # Remove rows with less than 5 ingredients
    filtered_data = filtered_data[filtered_data['Ingredients'].str.split(
        ',').str.len() >= 5]

    # Save the processed dataset
    filtered_data.to_csv(output_filepath, index=False)
    print(f"Processed data saved to {output_filepath}")


if __name__ == "__main__":
    input_filepath = "indianfooddatav2.xlsx"
    output_filepath = "preprocessed.csv"
    preprocess_data(input_filepath, output_filepath)
