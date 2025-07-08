from transformers import (
    EarlyStoppingCallback,
    PatchTSMixerConfig,
    PatchTSMixerForPrediction,
    PatchTSMixerForPretraining,
    Trainer,
    TrainingArguments,
)

def load_model(idx_cv, tool_age_predrilling=True):
    """

    :param idx_cv:
    :param tool_age_predrilling:
    :return:
    """
    if tool_age_predrilling:
        model_path = f'../trained_model/has_tool_age_predrilling/cv{idx_cv}'
        print(model_path)
        model = PatchTSMixerForPrediction.from_pretrained(
            model_path
        )
    else:
        model_path = f'../trained_model/no_tool_age_predrilling/cv{idx_cv}'
        model = PatchTSMixerForPrediction.from_pretrained(
            model_path
        )

    return model

