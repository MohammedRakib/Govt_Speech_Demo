python my-evaluation.py \
--model_id="openai/whisper-tiny" \
--dataset="google/fleurs" \
--config="bn_in" \
--language="bn" \
--split="test" \
--max_eval_samples="50" \
--batch_size="16" \
--do_bangla_unicode_normalize="True" \
--device="0" \
--streaming="False"
