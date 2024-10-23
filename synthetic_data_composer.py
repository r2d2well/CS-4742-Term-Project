import logging, random, json, os, copy, contextlib, sys, torch, pandas as pd
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer, MarianMTModel, MarianTokenizer
from tqdm import tqdm
from datasets import Dataset
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('omw-1.4')

# Remove all handlers associated with the root logger object
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# logs
file_handler = logging.FileHandler('debug.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Create a console handler to display warnings and errors in the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # Only show warnings and above
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def load_personality_data(file_path="data/personality_data.json"):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, file_path)

        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        logger.info(f"Successfully loaded personality data from '{json_path}'.")
        return data
    except FileNotFoundError:
        logger.error(f"File '{file_path}' not found. Please ensure the file exists in the specified directory.")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from '{file_path}': {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading '{file_path}': {e}")
    return {}

def initialize_paraphraser():

    #Initialize the paraphrasing pipeline using a T5 model with FP16 precision.
    logger.info("Initializing the paraphrasing pipeline...")
    try:
        # Using a T5 model fine-tuned for paraphrasing
        model_name = "ramsrigouthamg/t5_paraphraser"
        tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        model = T5ForConditionalGeneration.from_pretrained(model_name)

        # CUDA
        if torch.cuda.is_available():
            device = 0
            torch_dtype = torch.float16
        else:
            device = -1
            torch_dtype = torch.float32
            logger.info("CUDA is not available. Using CPU with FP32 precision for paraphraser.")

        paraphraser = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch_dtype,
            device=device,
            do_sample=True
        )
        logger.info("Paraphrasing pipeline initialized successfully.")
        return paraphraser
    except Exception as e:
        logger.error(f"Failed to initialize paraphrasing pipeline: {e}")
        raise

def initialize_back_translator():
    
    #Initialize the back-translation pipeline using MarianMT models with FP16 precision.
    
    logger.info("Initializing the back-translation pipeline...")
    try:
        # CUDA
        if torch.cuda.is_available():
            device = 0  # GPU 
            torch_dtype = torch.float16
        else:
            device = -1 # CPU
            torch_dtype = torch.float32
            logger.info("CUDA is not available. Using CPU with FP32 precision for back-translation.")

        # English to French
        model_name_en_fr = 'Helsinki-NLP/opus-mt-en-fr'
        tokenizer_en_fr = MarianTokenizer.from_pretrained(model_name_en_fr)
        model_en_fr = MarianMTModel.from_pretrained(model_name_en_fr)
        translator_en_fr = pipeline(
            "translation_en_to_fr",
            model=model_en_fr,
            tokenizer=tokenizer_en_fr,
            torch_dtype=torch_dtype,
            device=device
        )

        # French to English
        model_name_fr_en = 'Helsinki-NLP/opus-mt-fr-en'
        tokenizer_fr_en = MarianTokenizer.from_pretrained(model_name_fr_en)
        model_fr_en = MarianMTModel.from_pretrained(model_name_fr_en)
        translator_fr_en = pipeline(
            "translation_fr_to_en",
            model=model_fr_en,
            tokenizer=tokenizer_fr_en,
            torch_dtype=torch_dtype,
            device=device
        )

        logger.info("Back-translation pipeline initialized successfully.")
        return translator_en_fr, translator_fr_en
    except Exception as e:
        logger.error(f"Failed to initialize back-translation pipeline: {e}")
        raise

def paraphrase_text(paraphraser, texts, num_return_sequences=3, max_length=256, batch_size=16):
    input_texts = [f"paraphrase: {text}" for text in texts]
    try:
        # Create a Dataset from the texts
        dataset = Dataset.from_dict({'text': input_texts})
        
        with suppress_stdout():
            paraphrases = paraphraser(
                dataset['text'],
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                num_beams=5,
                temperature=1.5,
                early_stopping=True,
                batch_size=batch_size
            )

        # Process paraphrases as before
        paraphrased_texts = []
        for outputs in paraphrases:
            if isinstance(outputs, list):
                for output in outputs:
                    paraphrased_texts.append(output['generated_text'].strip())
            else:
                paraphrased_texts.append(outputs['generated_text'].strip())
        return paraphrased_texts
    except Exception as e:
        logger.error(f"Error during paraphrasing: {e}")
        return []

def synonym_replacement(text, n=2):
    
    # Replaces 'n' words in the sentence with their synonyms.
    
    words = text.split()
    new_words = copy.deepcopy(words)
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            synonym = synonym.replace("_", " ")
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return ' '.join(new_words)

def random_insertion(text, n=2):

    # Inserts 'n' amount of random synonyms into the sentence.
    
    words = text.split()
    new_words = copy.deepcopy(words)
    for _ in range(n):
        add_word = random.choice(words)
        synonyms = wordnet.synsets(add_word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            synonym = synonym.replace("_", " ")
            insert_idx = random.randint(0, len(new_words))
            new_words.insert(insert_idx, synonym)
    return ' '.join(new_words)

def back_translate(texts, translator_en_fr, translator_fr_en, batch_size=16):
    try:
        # Create a Dataset from the texts
        dataset = Dataset.from_dict({'text': texts})

        # Translate to French
        with suppress_stdout():
            translations_fr = translator_en_fr(
                dataset['text'],
                batch_size=batch_size
            )

        # Extract the French texts
        texts_fr = [t['translation_text'] for t in translations_fr]

        # Create a new Dataset for French texts
        dataset_fr = Dataset.from_dict({'text': texts_fr})

        # Translate back to English
        with suppress_stdout():
            translations_en = translator_fr_en(
                dataset_fr['text'],
                batch_size=batch_size
            )

        # Extract the back-translated English texts
        back_translated_texts = [t['translation_text'] for t in translations_en]
        return back_translated_texts
    except Exception as e:
        logger.error(f"Error during back-translation: {e}")
        return texts

def augment_text_batch(paraphraser, translator_en_fr, translator_fr_en, texts, batch_size=16):
    augmented_texts = set()

    # Paraphrasing
    paraphrases = paraphrase_text(paraphraser, texts, batch_size=batch_size)
    augmented_texts.update([p for p in paraphrases if p and p not in texts])

    # Back Translation
    back_translations = back_translate(texts, translator_en_fr, translator_fr_en, batch_size=batch_size)
    augmented_texts.update([bt for bt in back_translations if bt and bt not in texts])

    # Synonym Replacement and Random Insertion
    for text in texts:
        syn_repl = synonym_replacement(text)
        if syn_repl != text:
            augmented_texts.add(syn_repl)

        rand_insert = random_insertion(text)
        if rand_insert != text:
            augmented_texts.add(rand_insert)

    return list(augmented_texts)

def generate_synthetic_data(duplication_factors=None, output_csv="data/synthetic_personality_data.csv", data_file="data/personality_data.json", max_augmentations_per_post=4, checkpoint_interval=1000):
    """
    Generate synthetic data using multiple augmentation techniques and save to a CSV file.
    Creates checkpoints in a /results folder.
    """
    personality_data = load_personality_data(file_path=data_file)
    if not personality_data:
        logger.error("No personality data available. Exiting data generation.")
        return pd.DataFrame()

    # Create /results folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    logger.info("Starting synthetic data generation...")

    if duplication_factors is None:
        # Default to 20,000 records per personality type
        duplication_factors = {p_type: 20000 for p_type in personality_data.keys()}
        logger.info("No duplication factors provided. Using default of 20,000 records per personality type.")

    # Initialize augmentation pipelines
    paraphraser = initialize_paraphraser()
    translator_en_fr, translator_fr_en = initialize_back_translator()

    total_records_needed = sum(duplication_factors.values())

    # Load existing checkpoints and calculate total_unique_generated
    total_unique_generated = 0
    existing_generated_per_type = {}

    logger.info("Calculating existing progress from checkpoints...")
    for p_type, num_records_needed in duplication_factors.items():
        existing_checkpoints = [f for f in os.listdir(results_dir) if f.startswith(f"synthetic_data_{p_type}_checkpoint_") and f.endswith(".csv")]
        existing_generated = set()
        for checkpoint_file in existing_checkpoints:
            checkpoint_path = os.path.join(results_dir, checkpoint_file)
            df_checkpoint = pd.read_csv(checkpoint_path)
            existing_generated.update(df_checkpoint['posts'].tolist())
            
        # Limit the existing generated data to the number of records needed
        existing_generated = set(list(existing_generated)[:num_records_needed])
        existing_generated_per_type[p_type] = existing_generated
        total_unique_generated += len(existing_generated)

    total_unique_generated = min(total_unique_generated, total_records_needed)

    logger.info(f"Total records already generated: {total_unique_generated}")

    # Prepare total progress bar with initial value
    with tqdm(total=total_records_needed, desc="Total Progress", leave=True, initial=total_unique_generated) as total_pbar:
        for p_type, num_records in duplication_factors.items():
            posts = personality_data.get(p_type)
            if not posts:
                logger.warning(f"No posts found for personality type '{p_type}'. Skipping...")
                continue
            logger.info(f"Generating {num_records} records for personality type '{p_type}'.")

            # Initialize checkpoint variables
            synthetic_data_p_type = []
            checkpoint_counter = 0

            # Use existing_generated_per_type to get existing generated data
            unique_generated = existing_generated_per_type.get(p_type, set())
            existing_posts = set(posts)
            total_needed = num_records

            initial_progress = min(len(unique_generated), total_needed)

            batch_size = 32  # Adjust batch size based on GPU memory

            # Initialize per-type progress bar
            pbar_desc = f"Generating for {p_type}"
            with tqdm(total=total_needed, desc=pbar_desc, leave=False, initial=initial_progress) as pbar:
                if initial_progress >= total_needed:
                    pbar.update(total_needed - pbar.n)
                    continue
                while len(unique_generated) < total_needed:
                    # Collect a batch of base posts
                    base_posts = random.choices(posts, k=batch_size)
                    
                    # Apply augmentation
                    augmented_texts = augment_text_batch(paraphraser, translator_en_fr, translator_fr_en, base_posts, batch_size=batch_size)
                    for augmented_post in augmented_texts:
                        augmented_post = augmented_post.strip()
                        if augmented_post and augmented_post not in existing_posts and augmented_post not in unique_generated:
                            synthetic_data_p_type.append({"posts": augmented_post, "type": p_type})
                            unique_generated.add(augmented_post)
                            total_unique_generated += 1
                            pbar.update(1)
                            total_pbar.update(1)
                            if len(unique_generated) >= total_needed:
                                break
                            
                    # Save checkpoint if interval reached
                    if len(synthetic_data_p_type) >= checkpoint_interval:
                        # Save to checkpoint file
                        checkpoint_filename = f"synthetic_data_{p_type}_checkpoint_{checkpoint_counter}.csv"
                        checkpoint_path = os.path.join(results_dir, checkpoint_filename)
                        df_checkpoint = pd.DataFrame(synthetic_data_p_type)
                        df_checkpoint.to_csv(checkpoint_path, index=False)
                        # Reset synthetic_data_p_type
                        synthetic_data_p_type = []
                        checkpoint_counter += 1
                        
                    # Break the loop if no new augmentations can be generated
                    if len(augmented_texts) == 0:
                        logger.warning(f"Could not generate more unique augmentations for personality type '{p_type}'.")
                        break
                    
                # Save any remaining data for this personality type
                if synthetic_data_p_type:
                    checkpoint_filename = f"synthetic_data_{p_type}_checkpoint_{checkpoint_counter}.csv"
                    checkpoint_path = os.path.join(results_dir, checkpoint_filename)
                    df_checkpoint = pd.DataFrame(synthetic_data_p_type)
                    df_checkpoint.to_csv(checkpoint_path, index=False)
                    logger.info(f"Saved final checkpoint for {p_type} to {checkpoint_filename}")

    logger.info("Loading existing data to include in the CSV.")
    existing_data = []
    for p_type, posts in personality_data.items():
        for post in posts:
            existing_data.append({"posts": post, "type": p_type})

    # Load all synthetic data from checkpoints
    logger.info("Loading synthetic data from checkpoints.")
    synthetic_data_frames = []
    for filename in os.listdir(results_dir):
        if filename.startswith("synthetic_data_") and filename.endswith(".csv"):
            checkpoint_path = os.path.join(results_dir, filename)
            df_checkpoint = pd.read_csv(checkpoint_path)
            synthetic_data_frames.append(df_checkpoint)
    if synthetic_data_frames:
        synthetic_df = pd.concat(synthetic_data_frames, ignore_index=True)
        logger.info(f"Total synthetic records loaded from checkpoints: {synthetic_df.shape[0]}")
    else:
        synthetic_df = pd.DataFrame()
        logger.warning("No synthetic data found in checkpoints.")

    # Create DataFrame for existing data
    existing_df = pd.DataFrame(existing_data)
    logger.info(f"Total existing records loaded: {existing_df.shape[0]}")

    # Concatenate existing and synthetic data
    combined_df = pd.concat([existing_df, synthetic_df], ignore_index=True)

    # Shuffle the combined DataFrame
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)

    # Save to CSV if output_csv is specified
    if output_csv:
        combined_df.to_csv(output_csv, columns=["posts", "type"], index=False)
        logger.info(f"Synthetic data generation completed. Combined data saved to '{output_csv}'.")
    else:
        logger.info("Synthetic data generation completed. No output CSV file specified.")

    # Delete checkpoint files after successful CSV creation
    logger.info("Deleting checkpoint files...")
    for filename in os.listdir(results_dir):
        if filename.startswith("synthetic_data_") and filename.endswith(".csv"):
            file_path = os.path.join(results_dir, filename)
            try:
                os.remove(file_path)
                logger.info(f"Deleted checkpoint file: {filename}")
            except Exception as e:
                logger.error(f"Error deleting file {filename}: {e}")

    logger.info("All checkpoint files have been deleted.")

    return combined_df

if __name__ == "__main__":
    generate_synthetic_data()