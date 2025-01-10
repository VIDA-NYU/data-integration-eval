from magneto import Magneto
import pandas as pd

if __name__ == "__main__":
    # given
    source = pd.DataFrame({"column_1": ["a1", "b1", "c1"], "col_2": ["a2", "b2", "c2"]})
    target = pd.DataFrame({"column_1a": ["a1", "b1", "c1"], "col2": ["a2", "b2", "c2"]})

    # when
    mode = "header_values_verbose"
    model_path = (
        "/Users/rlopez/Downloads/mpnet-gdc-header_values_verbose-semantic-64-0.5.pth"
    )

    mag = Magneto(encoding_mode=mode, embedding_model=model_path)
    matches = mag.get_matches(source, target)

    print(matches)
