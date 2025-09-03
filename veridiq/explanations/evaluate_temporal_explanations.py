import click
import numpy as np

from veridiq.show_predictions import (
    SUBSAMPLING_FACTORS,
    eval_per_video,
    get_predictions_path,
    load_test_metadata,
    select_fvfa,
)


def evaluate_temporal_explanations(feature_extractor_type):
    preds = np.load(get_predictions_path(feature_extractor_type), allow_pickle=True)
    metadata = load_test_metadata(feature_extractor_type)

    preds, metadata = select_fvfa(preds, metadata)
    scores_video = eval_per_video(
        preds,
        metadata,
        feature_extractor_type=feature_extractor_type,
        to_binarize=False,
    )
    print(len(scores_video))

    scores_video = [s for s in scores_video if s is not None]
    print(len(scores_video))

    score = 100 * np.mean(scores_video)
    print("{:.2f}".format(score))


@click.command()
@click.option("-f", "--feature", "feature_extractor_type", type=str)
def main(feature_extractor_type):
    evaluate_temporal_explanations(feature_extractor_type)


if __name__ == "__main__":
    main()
