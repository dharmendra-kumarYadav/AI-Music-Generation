def transpose(sequence, semitones):
    """
    Transpose a sequence of notes up or down by a given number of semitones.

    Args:
    - sequence: List of notes to be transposed.
    - semitones: Number of semitones to transpose. Positive value transposes up, negative value transposes down.

    Returns:
    - transposed_sequence: List of transposed notes.
    """
    # Mapping of notes to their positions in the chromatic scale
    chromatic_scale = {
        'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
        'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
    }

    # Transpose each note in the sequence
    transposed_sequence = []
    for note in sequence:
        if note in chromatic_scale:
            original_position = chromatic_scale[note]
            transposed_position = (original_position + semitones) % 12
            transposed_note = [key for key, value in chromatic_scale.items() if value == transposed_position][0]
            transposed_sequence.append(transposed_note)
        else:
            # If the note is not found in the chromatic scale, keep it unchanged
            transposed_sequence.append(note)

    return transposed_sequence
