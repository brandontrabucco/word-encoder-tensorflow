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


    # Store encoded lengths and columns
    encoded_columns = []
    actual_lengths = []


    # Iterate for every column
    for input_characters in input_characters_columns:

        # Query lookup table
        one_hot_tensor = tf.one_hot(lookup_table.lookup(input_characters.values), len(vocabulary), dtype=tf.float32)


        # Calculate current sequence length
        current_length = tf.size(one_hot_tensor) // len(vocabulary)


        # Pad input to match DATASET_MAXIMUM
        expanded_column = tf.pad(one_hot_tensor, [[0, (DATASET_MAXIMUM - current_length)], [0, 0]])


        # Store one-hot vector as encoded column
        encoded_columns.append(expanded_column)


        # Store original length of sequence
        actual_lengths.append(current_length)

    return encoded_columns, actual_lengths


# Read single row words
def decode_record_wikipedia(filename_queue, num_columns=DATASET_COLUMNS, vocabulary=DATASET_VOCABULARY, default_value=DATASET_DEFAULT):

    # Attach text file reader
    DATASET_READER = tf.TextLineReader()


    # Read single line from dataset
    key, value_text = DATASET_READER.read(filename_queue)


    # Decode line to columns 
    value_columns = tf.decode_csv(value_text, [[default_value] for i in range(num_columns)])


    # Convert characters to sparse tensor
    encoded_columns, actual_lengths = columns_to_sparse_tensor(value_columns)

    return tf.reshape(tf.stack(encoded_columns), [num_columns, DATASET_MAXIMUM, len(vocabulary)]), tf.stack(actual_lengths)


# Batch configuration constants
BATCH_SIZE = 32
NUM_THREADS = 4
TOTAL_EXAMPLES = 500000


# Generate batch from rows
def generate_batch(example, actual_length, batch_size=BATCH_SIZE, num_threads=NUM_THREADS, shuffle_batch=True):

    # Shuffle batch randomly
    if shuffle_batch:

        # Construct batch from queue of records
        example_batch, length_batch = tf.train.shuffle_batch(
            [example, actual_length],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=TOTAL_EXAMPLES,
            min_after_dequeue=(TOTAL_EXAMPLES // 50))


    # Preserve order of batch
    else:

        # Construct batch from queue of records
        example_batch, length_batch = tf.train.batch(
            [example, actual_length],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=TOTAL_EXAMPLES)

    return example_batch, length_batch



# Generate single training batch of examples wikipedia
def training_batch_wikipedia():

    # A queue to generate batches
    filename_queue = tf.train.string_input_producer(DATASET_FILENAMES_WIKIPEDIA)


    # Decode from string to floating point
    example, actual_length = decode_record_wikipedia(filename_queue)


    # Combine example queue into batch
    example_batch, length_batch = generate_batch(example, actual_length)

    return tf.transpose(example_batch, [0, 2, 3, 1]), length_batch


# Naming conventions
PREFIX_RNN = "rnn"
PREFIX_DENSE = "dense"
PREFIX_SOFTMAX = "softmax"
PREFIX_TOTAL = "total"


# Naming conventions
EXTENSION_NUMBER = (lambda number: "_" + str(number))
EXTENSION_LOSS = "_loss"
EXTENSION_WEIGHTS = "_weights"
EXTENSION_BIASES = "_biases"
EXTENSION_OFFSET = "_offset"
EXTENSION_SCALE = "_scale"
EXTENSION_ACTIVATION = "_activation"
EXTENSION_COLUMN = "_column"


# Naming conventions
COLLECTION_LOSSES = "losses"
COLLECTION_PARAMETERS = "parameters"
COLLECTION_ACTIVATIONS = "activations"


# Initialize trainable parameters
def initialize_weights_cpu(name, shape, standard_deviation=0.01, decay_factor=None):

    # Force usage of cpu
    with tf.device("/cpu:0"):

        # Sample weights from normal distribution
        weights = tf.get_variable(
            name,
            shape, 
            initializer=tf.truncated_normal_initializer(
                stddev=standard_deviation,
                dtype=tf.float32),
            dtype=tf.float32)

    # Add weight decay to loss function
    if decay_factor is not None:

        # Calculate decay with l2 loss
        weight_decay = tf.multiply(
            tf.nn.l2_loss(weights), 
            decay_factor, 
            name=(name + EXTENSION_LOSS))
        tf.add_to_collection(COLLECTION_LOSSES, weight_decay)

    return weights


# Initialize trainable parameters
def initialize_biases_cpu(name, shape):

    # Force usage of cpu
    with tf.device("/cpu:0"):

        # Sample weights from normal distribution
        biases = tf.get_variable(
            name,
            shape, 
            initializer=tf.constant_initializer(1.0),
            dtype=tf.float32)

    return biases


def inference_skipgram_wikipedia(example_batch, length_batch):

    # Bind to name for consistency
    context_word = tf.concat([example_batch[:, :, :, :(DATASET_COLUMNS // 2)], example_batch[:, :, :, (DATASET_COLUMNS // 2 + 1):]], 3)
    context_length = tf.concat([length_batch[:, :(DATASET_COLUMNS // 2)], length_batch[:, (DATASET_COLUMNS // 2 + 1):]], 1)


    # Extract centeroid word
    cenetroid_word = example_batch[:, :, :, (DATASET_COLUMNS // 2 + 1)]
    centroid_length = length_batch[:, (DATASET_COLUMNS // 2 + 1)]

    
    # Create lstm cells with double vocabulary size
    lstm_forward = tf.contrib.rnn.LSTMCell((len(DATASET_VOCABULARY) * 2), use_peepholes=True)
    lstm_backward = tf.contrib.rnn.LSTMCell((len(DATASET_VOCABULARY) * 2), use_peepholes=True)


    # Process each column separately
    for i in range(DATASET_COLUMNS - 1):
        
        # Create scope for first rnn layer
        with tf.variable_scope(PREFIX_RNN + EXTENSION_NUMBER(1) + EXTENSION_COLUMN + EXTENSION_NUMBER(i)) as scope:

            # Compute recurrent activation
            activation, state = tf.nn.bidirectional_dynamic_rnn(
                lstm_forward, 
                lstm_backward, 
                context_word[:, :, :, i], 
                sequence_length=context_length[:, i],
                dtype=tf.float32,
                scope=scope)


            # Cocatenate tuple to tensor
            activation = tf.reshape(tf.concat(activation, 2), [BATCH_SIZE, -1])


        # Create scope for first linear layer
        with tf.variable_scope(PREFIX_DENSE + EXTENSION_NUMBER(1) + EXTENSION_COLUMN + EXTENSION_NUMBER(i)) as scope:

            # Create matrix of weights
            weights = initialize_weights_cpu((scope.name + EXTENSION_WEIGHTS), [(len(DATASET_VOCABULARY) * 4 * DATASET_MAXIMUM), 512])


            # Multiple weight matrix by activation
            activation = tf.matmul(activation, weights)
        

            # Create  and add biases
            biases = initialize_biases_cpu((scope.name + EXTENSION_BIASES), [512])
            activation = tf.nn.bias_add(activation, biases)


    # Note the final layer has no activation function
    return activation


# Create new graph
with tf.Graph().as_default():

    batch, length = training_batch_wikipedia()
    inference = inference_skipgram_wikipedia(batch, length)

    with tf.train.MonitoredTrainingSession() as session:

        output = session.run(inference)
        print(output.shape)