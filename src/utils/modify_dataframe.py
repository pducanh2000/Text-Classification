def get_essay(essay_fns):
    essay_cache = {}

    output = []
    for essay_fn in essay_fns:
        if essay_fn not in list(essay_cache.keys()):
            essay_txt = open(essay_fn).read().strip().lower()
            essay_cache[essay_fn] = essay_txt
        output.append(essay_cache[essay_fn])

    return output


def general_modify_dataframe(df, basepath, tokenizer):
    # df['inputs'] = df.discourse_type.str.lower() + ' ' + tokenizer.sep_token + ' ' + df.topic_name + ' ' +
    # tokenizer.sep_token + ' ' + df.discourse_text.str.lower()
    df['inputs'] = df.discourse_type.str.lower() + ' ' + tokenizer.sep_token + ' ' + df.discourse_text.str.lower()

    # Create essays column
    # df["essay_fns"] = basepath + '/' + df.essay_id + '.txt'
    list_essay_filepath = basepath + '/' + df.essay_id + '.txt'
    df['essays'] = get_essay(list_essay_filepath)
    return df


def create_numerical_label(df, cfg):
    df["labels"] = df["discourse_effectiveness"]
    numerical_label = {"labels": cfg["label_map"]}
    df.replace(numerical_label, inplace=True)
    return df
