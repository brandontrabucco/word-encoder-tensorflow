import tensorflow as tf


# The location on the disk of project
PROJECT_BASEDIR = ("C:/Users/brand/Google Drive/" +
    "Academic/Research/Deep Recurrent Language Vectorization/")


# The location on the disk of checkpoints
CHECKPOINT_BASEDIR = (PROJECT_BASEDIR + "Checkpoints/")


# The location on the disk of wikipedia dataset csv
DATASET_BASEDIR_WIKIPEDIA = ("C:/Users/brand/Google Drive/" +
    "Academic/Research/Datasets/wikipedia/")


# Filenames associated with wikipedia
DATASET_FILENAMES_WIKIPEDIA= [
    (DATASET_BASEDIR_WIKIPEDIA + "grams_9.csv")
]


# Locate dataset files on hard disk
for FILE_WIKIPEDIA in DATASET_FILENAMES_WIKIPEDIA:
    if not tf.gfile.Exists(FILE_WIKIPEDIA):
        raise ValueError('Failed to find file: ' + FILE_WIKIPEDIA)


# Dataset configuration constants
DATASET_COLUMNS = 9
DATASET_MAXIMUM = 50
DATASET_VOCABULARY = "abcdefghijklmnopqrstuvwxyz"
DATASET_DEFAULT = "a"


# Convert words into one-hot tensor
def columns_to_sparse_tensor(dataset_columns, vocabulary=DATASET_VOCABULARY):

    
    # List allowed characters
    mapping_characters = tf.string_split([vocabulary], delimiter="")


    # List characters in each word
    input_characters_columns = [tf.string_split([column], delimiter="") for column in dataset_columns]


    # Convert integer lookup table
    lookup_table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_characters.values, default_value=0)


    # Encode words into one-hot tensor
    encoded_columns = []
    for input_characters in input_characters_columns:
        one_hot_tensor = tf.one_hot(lookup_table.lookup(input_characters.values), len(vocabulary), dtype=tf.int8)
        current_length = tf.size(one_hot_tensor) // len(vocabulary)
        expanded_column = tf.pad(one_hot_tensor, [[0, (DATASET_MAXIMUM - current_length)], [0, 0]])
        encoded_columns.append(expanded_column)

    return encoded_columns


# Read single row words
def decode_record_wikipedia(filename_queue, num_columns=DATASET_COLUMNS, vocabulary=DATASET_VOCABULARY, default_value=DATASET_DEFAULT):

    # Attach text file reader
    DATASET_READER = tf.TextLineReader()


    # Read single line from dataset
    key, value_text = DATASET_READER.read(filename_queue)


    # Decode line to columns 
    value_columns = tf.decode_csv(value_text, [[default_value] for i in range(num_columns)])


    # Convert characters to sparse tensor
    value_sparse = tf.stack(columns_to_sparse_tensor(value_columns))

    return tf.reshape(value_sparse, [num_columns, DATASET_MAXIMUM, len(vocabulary)])


# Batch configuration constants
BATCH_SIZE = 32
NUM_THREADS = 4
TOTAL_EXAMPLES = 500000


# Generate batch from rows
def generate_batch(example, batch_size=BATCH_SIZE, num_threads=NUM_THREADS, shuffle_batch=True):

    # Shuffle batch randomly
    if shuffle_batch:

        # Construct batch from queue of records
        example_batch = tf.train.shuffle_batch(
            [example],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=TOTAL_EXAMPLES,
            min_after_dequeue=(TOTAL_EXAMPLES // 50))


    # Preserve order of batch
    else:

        # Construct batch from queue of records
        example_batch = tf.train.batch(
            [example],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=TOTAL_EXAMPLES)

    return example_batch


# Create new graph
with tf.Graph().as_default():

    filename_queue = tf.train.string_input_producer(DATASET_FILENAMES_WIKIPEDIA)
    dataset_record = decode_record_wikipedia(filename_queue)
    dataset_batch = generate_batch(dataset_record)

    with tf.train.MonitoredTrainingSession() as session:

        output = session.run(dataset_batch)
        print(output.shape)