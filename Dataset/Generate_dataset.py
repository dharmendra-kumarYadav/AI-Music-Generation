import numpy as np
from music21 import *
from collections import Counter

def get_sequences(notes_array, timesteps=32, future_steps=8):
    # Extract all individual notes from the list of MIDI notes
    notes_ = [element for note_ in notes_array for element in note_]
    
    # Count the occurrences of each note
    freq = dict(Counter(notes_))
    
    # Filter out less frequent notes based on a threshold (e.g., 50 occurrences)
    frequent_notes = [note_ for note_, count in freq.items() if count >= 50]
    
    new_music = []

    # Filter the notes_array to include only frequent notes
    for notes in notes_array:
        temp = [note_ for note_ in notes if note_ in frequent_notes]
        new_music.append(temp)
        
    new_music = np.array(new_music, dtype=object)

    no_of_timesteps = timesteps
    x = []
    y = []
    
    # Generate input-output pairs with overlapping sequences
    for note_ in new_music:
        for i in range(0, len(note_) - (no_of_timesteps + future_steps), 1):
            # Prepare input and output sequences
            input_ = note_[i:i + no_of_timesteps]
            output = note_[i + 1 + no_of_timesteps:i + 1 + no_of_timesteps + future_steps]
            
            x.append(input_)
            y.append(output)
            
    x = np.array(x)
    y = np.array(y)

    # Prepare input sequence by assigning unique integers to notes
    unique_notes = list(set(x.ravel()))
    note_to_int = dict((note_, number) for number, note_ in enumerate(unique_notes))
    x_seq = [[note_to_int[note_] for note_ in seq] for seq in x]
    x_seq = np.array(x_seq)

    # Prepare output sequence similarly
    y_seq = [[note_to_int[note_] for note_ in seq] for seq in y]
    y_seq = np.array(y_seq)

    # Print unique notes for debugging or analysis
    print(unique_notes)

    return x_seq, y_seq, unique_notes, note_to_int
